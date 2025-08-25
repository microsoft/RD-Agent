"""Context7 MCP integration with enhanced multi-round tool calling.

This module provides an improved interface for querying documentation
using MCP (Model Context Protocol) tools with support for sequential tool calls.
"""

import asyncio
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def _is_documentation_not_found(result_text: str) -> bool:
    """Check if result indicates documentation not found."""
    error_indicators = ["Failed to fetch documentation", "Error code: 404", "Documentation not found", "404 Not Found"]
    return any(indicator in result_text for indicator in error_indicators)


def _is_connection_error(result_text: str) -> bool:
    """Check if result indicates connection issues."""
    connection_indicators = ["Connection failed", "Timeout", "Network error", "Service unavailable"]
    return any(indicator in result_text for indicator in connection_indicators)


from rdagent.components.mcp.cache import get_mcp_cache
from rdagent.components.mcp.conf import get_mcp_global_settings
from rdagent.components.mcp.context7.conf import (
    get_context7_settings,
    is_context7_enabled,
)
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T


class MCPOpenAIClient:
    """Enhanced client for interacting with OpenAI models using MCP tools."""

    def __init__(
        self, model: str = "gpt-4.1", server_url: str = "", api_key: str = "", api_base: str = "", verbose: bool = False
    ):
        """Initialize the OpenAI MCP client.

        Args:
            model: The OpenAI model to use
            server_url: The URL of the MCP server
            api_key: OpenAI API key
            api_base: OpenAI API base URL
            verbose: Enable verbose logging for detailed debugging
        """
        self.session: Optional[ClientSession] = None
        self.model = model
        self.server_url = server_url
        self.verbose = verbose

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

        # Get available MCP tools
        tools = await self.get_mcp_tools()
        # Initialize conversation
        messages = [{"role": "user", "content": query}]

        for round_count in range(1, max_rounds + 1):
            if self.verbose:
                logger.info(f"Round {round_count}: Calling OpenAI...")

            response = await self._call_openai_api(messages, tools)
            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            if not assistant_message.tool_calls:
                logger.info(f"Final response received in round {round_count}")
                return assistant_message.content

            await self._process_tool_calls(assistant_message.tool_calls, messages)
            if self.verbose:
                logger.info(f"Round {round_count} completed, checking if more tools needed...")

        logger.warning(f"Reached maximum rounds ({max_rounds}), returning last response")
        return messages[-1].content if messages else "No response generated"

    async def _call_openai_api(self, messages: List[Dict], tools: List[Dict[str, Any]]):
        """Call OpenAI API with current messages and available tools."""
        return await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

    async def _process_tool_calls(self, tool_calls, messages: List[Dict]) -> None:
        """Process all tool calls and add results to messages."""
        tool_names = [tool_call.function.name for tool_call in tool_calls]
        if self.verbose:
            logger.info(f"Executing {len(tool_calls)} tool(s): {', '.join(tool_names)}")

        for i, tool_call in enumerate(tool_calls, 1):
            if self.verbose:
                logger.info(f"Tool {i}: {tool_call.function.name}")
            result_message = await self._execute_single_tool(tool_call, i)
            messages.append(result_message)

    async def _execute_single_tool(self, tool_call, tool_index: int) -> Dict[str, str]:
        """Execute a single tool call and return the result message."""
        try:
            result = await self.session.call_tool(
                tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
            )
            if self.verbose:
                logger.info(f"Tool {tool_index} executed successfully")

            result_text = result.content[0].text
            content = self._handle_tool_result(result_text, tool_index)

            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": content,
            }
        except Exception as e:
            logger.error(f"Tool {tool_index} failed: {str(e)}")
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"Error executing tool: {str(e)}",
            }

    def _handle_tool_result(self, result_text: str, tool_index: int) -> str:
        """Handle tool result with simple error detection."""
        if _is_documentation_not_found(result_text):
            logger.warning(f"Documentation not found for requested library")
            return (
                "Documentation not found for this library. This library may not have detailed "
                "documentation available in the knowledge base, but you can still provide general "
                "guidance based on the library information from resolve-library-id."
            )

        if _is_connection_error(result_text):
            logger.warning(f"Connection error while fetching documentation")
            return "Connection error occurred while fetching documentation. Please try again later."

        return result_text

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
                client = MCPOpenAIClient(
                    model=model, server_url=mcp_http_url, api_key=api_key, api_base=api_base, verbose=verbose
                )
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

                # Log the final result
                logger.info(f"Context7 query completed successfully in {total_time:.2f}s")

                # Log result based on verbosity and length
                if verbose:
                    # In verbose mode, show full result
                    logger.info(f"Context7 result: {response}")
                else:
                    # In normal mode, show preview if result is long
                    if len(response) > 300:
                        result_preview = response[:300] + "..."
                        logger.info(f"Context7 result preview: {result_preview}")
                    else:
                        logger.info(f"Context7 result: {response}")

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
