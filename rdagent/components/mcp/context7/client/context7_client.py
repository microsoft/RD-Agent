"""Context7 MCP integration with enhanced multi-round tool calling.

This module provides an improved interface for querying documentation
using MCP (Model Context Protocol) tools with support for sequential tool calls.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import nest_asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rdagent.components.mcp.cache import get_mcp_cache
from rdagent.components.mcp.conf import get_mcp_global_settings
from rdagent.components.mcp.context7.conf import (
    get_context7_settings,
    is_context7_enabled,
)
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()


class MCPOpenAIClient:
    """Enhanced client for interacting with OpenAI models using MCP tools."""

    def __init__(self, model: str = "gpt-4.1", server_url: str = "", api_key: str = "", api_base: str = ""):
        """Initialize the OpenAI MCP client.

        Args:
            model: The OpenAI model to use
            server_url: The URL of the MCP server
            api_key: OpenAI API key
            api_base: OpenAI API base URL
        """
        self.session: Optional[ClientSession] = None
        self.model = model
        self.server_url = server_url

        self.openai_client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server in OpenAI format.

        Returns:
            A list of tools in OpenAI format
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Call connect() first.")

        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]

    async def process_query(self, query: str, max_rounds: int = 5) -> str:
        """Process a query using OpenAI and available MCP tools with multiple rounds.

        Args:
            query: The user query
            max_rounds: Maximum number of tool calling rounds to prevent infinite loops

        Returns:
            The response from OpenAI
        """
        # Get available tools
        tools = await self.get_mcp_tools()

        # Initialize conversation
        messages = [{"role": "user", "content": query}]

        round_count = 0

        while round_count < max_rounds:
            round_count += 1
            logger.info(f"Round {round_count}: Calling OpenAI...")

            # Check if we need to force get-library-docs call
            tool_choice = "auto"

            # Let AI decide based on the prompt instructions
            # The prompt now explicitly instructs AI to call get-library-docs for each resolve-library-id

            # Call OpenAI API
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )

            # Get assistant's response
            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            # Check if there are tool calls
            if assistant_message.tool_calls:
                logger.info(f"Processing {len(assistant_message.tool_calls)} tool call(s)...")

                # Process each tool call
                for i, tool_call in enumerate(assistant_message.tool_calls, 1):
                    logger.info(f"Tool {i}: {tool_call.function.name}")

                    try:
                        # Execute tool call
                        result = await self.session.call_tool(
                            tool_call.function.name,
                            arguments=json.loads(tool_call.function.arguments),
                        )
                        logger.info(f"Tool {i} executed successfully")

                        # Check if the result contains error information
                        result_text = result.content[0].text
                        if "Failed to fetch documentation" in result_text or "Error code: 404" in result_text:
                            logger.warning(f"Tool {i} returned 404 error - library documentation not found")
                            # Add a helpful error message instead of the raw error
                            error_msg = f"Documentation not found for this library. This library may not have detailed documentation available in the knowledge base, but you can still provide general guidance based on the library information from resolve-library-id."
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": error_msg,
                                }
                            )
                        else:
                            # Add successful tool response to conversation
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": result_text,
                                }
                            )
                    except Exception as e:
                        logger.error(f"Tool {i} failed: {str(e)}")
                        # Add error response to conversation
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error executing tool: {str(e)}",
                            }
                        )

                # Continue to next round to let AI decide if it needs more tools
                logger.info(f"Round {round_count} completed, checking if more tools needed...")
                continue
            else:
                # No tool calls, we have the final response
                logger.info(f"Final response received in round {round_count}")
                return assistant_message.content

        # If we've reached max rounds, return the last response
        logger.warning(f"Reached maximum rounds ({max_rounds}), returning last response")
        return messages[-1].content if messages else "No response generated"

    async def cleanup(self):
        """Clean up resources."""
        # Session cleanup is handled by the context manager
        pass


@retry(
    stop=stop_after_attempt(3),  # Retry 2 times, total 3 attempts
    wait=wait_exponential(multiplier=1, min=3, max=20),  # Exponential backoff: 3s, 6s, 12s
    retry=retry_if_exception_type(
        (
            ConnectionError,
            TimeoutError,
            RuntimeError,
            OSError,
        )
    ),
)
async def _query_context7_with_retry(
    error_message: str, full_code: Optional[str] = None, max_rounds: int = 5, verbose: bool = False
) -> Optional[str]:
    """Internal function with retry mechanism for Context7 enhanced queries.

    Args:
        error_message: The error message or traceback to search for
        full_code: Complete code context for better understanding (optional)
        max_rounds: Maximum number of tool calling rounds
        verbose: Enable verbose logging (default: False)

    Returns:
        Documentation search result as string, or None if failed
    """
    # Load configuration using pydantic settings
    settings = get_context7_settings()
    mcp_http_url = settings.mcp_url
    model = settings.model
    api_key = settings.api_key
    api_base = settings.api_base

    # Check cache setting
    global_settings = get_mcp_global_settings()
    cache = get_mcp_cache() if global_settings.cache_enabled else None
    if cache:
        cached_result = cache.get_query_result(error_message)
        if cached_result:
            logger.info("Returning cached query result")
            cache.log_cache_stats()
            return cached_result

    # Record start time for execution timing
    start_time = time.time()

    try:
        # Connect to the server using Streamable HTTP
        # Use a fixed URL for streamable HTTP connection

        async with streamablehttp_client(mcp_http_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the connection
                await session.initialize()

                # List available tools
                if verbose:
                    tools_result = await session.list_tools()
                    logger.info("Available tools:")
                    for tool in tools_result.tools:
                        logger.info(f"  - {tool.name}: {tool.description}")

                # Create OpenAI client and process query
                client = MCPOpenAIClient(model=model, server_url=mcp_http_url, api_key=api_key, api_base=api_base)
                client.session = session  # Use the existing session

                # Build context information using template
                context_info = ""
                if full_code:
                    context_info = T("rdagent.components.mcp.context7.prompts.templates:code_context_template").r(
                        full_code=full_code
                    )

                # ADD SPECIAL CASE FOR TIMM LIBRARY
                timm_trigger = error_message.lower().count("timm") >= 3
                timm_trigger_text = ""
                if timm_trigger:
                    timm_trigger_text = T("rdagent.components.mcp.context7.prompts.templates:timm_special_case").r()

                # Construct enhanced query using template
                enhanced_query = T(
                    "rdagent.components.mcp.context7.prompts.templates:context7_enhanced_query_template"
                ).r(error_message=error_message, context_info=context_info, timm_trigger_text=timm_trigger_text)

                # Process the enhanced query with multi-round tool calling
                response = await client.process_query(enhanced_query, max_rounds=max_rounds)

                # Calculate and display total execution time
                end_time = time.time()
                total_time = end_time - start_time
                logger.info(f"Context7 enhanced total execution time: {total_time:.2f} seconds")
                logger.info(
                    f"Context7 enhanced execution completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
                )

                result = response

                # Cache the query result
                if cache:
                    cache.set_query_result(error_message, result)
                    cache.log_cache_stats()

                return result

    except Exception as e:
        logger.error(f"Error in Context7 enhanced query: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        if hasattr(e, "__cause__") and e.__cause__:
            logger.error(f"Caused by: {e.__cause__}")
        raise


async def query_context7(
    error_message: str, full_code: Optional[str] = None, max_rounds: int = 5, verbose: bool = False
) -> Optional[str]:
    """Query context7 documentation with enhanced multi-round tool calling and retry mechanism.

    Args:
        error_message: The error message or traceback to search for
        full_code: Complete code context for better understanding (optional)
        max_rounds: Maximum number of tool calling rounds (default: 5)
        verbose: Enable verbose logging (default: False)

    Returns:
        Documentation search result as string, or None if failed
    """
    # Check if Context7 is enabled and available

    if not is_context7_enabled():
        logger.warning("Context7 MCP is disabled. Check MCP global settings and Context7 configuration.")
        return None
    try:
        return await _query_context7_with_retry(error_message, full_code, max_rounds, verbose)
    except (ConnectionError, TimeoutError, RuntimeError, OSError) as e:
        # These are retryable errors, but retries have failed
        logger.error(f"Context7 enhanced query failed after retries due to {type(e).__name__}: {str(e)}")
        return None
    except Exception as e:
        # Other non-retryable errors (e.g., configuration errors, authentication failures)
        logger.error(f"Context7 enhanced query failed due to non-retryable error {type(e).__name__}: {str(e)}")
        return None
