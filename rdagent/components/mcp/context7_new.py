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
from rdagent.components.mcp.util import get_context7_settings
from rdagent.log import rdagent_logger as logger

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()


class MCPOpenAIClient:
    """Enhanced client for interacting with OpenAI models using MCP tools."""

    def __init__(self, model: str = "gpt-4o", server_url: str = "http://localhost:8123/mcp"):
        """Initialize the OpenAI MCP client.

        Args:
            model: The OpenAI model to use
            server_url: The URL of the MCP server
        """
        self.session: Optional[ClientSession] = None
        self.model = model
        self.server_url = server_url

        # Get settings for OpenAI configuration
        settings = get_context7_settings()
        self.openai_client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.api_base)

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

            # Check if the previous round called resolve-library-id
            if len(messages) >= 2:
                # Look for the last assistant message with tool calls
                for msg in reversed(messages):
                    # Handle both dict messages and ChatCompletionMessage objects
                    if isinstance(msg, dict):
                        role = msg.get("role")
                        tool_calls = msg.get("tool_calls")
                    else:
                        role = getattr(msg, "role", None)
                        tool_calls = getattr(msg, "tool_calls", None)

                    if role == "assistant" and tool_calls:
                        # Check if any of the last tool calls was resolve-library-id
                        if any(tc.function.name == "resolve-library-id" for tc in tool_calls):
                            # Check if get-library-docs has NOT been called yet
                            has_called_get_docs = False
                            for m in messages:
                                if isinstance(m, dict):
                                    m_role = m.get("role")
                                    m_tool_calls = m.get("tool_calls")
                                else:
                                    m_role = getattr(m, "role", None)
                                    m_tool_calls = getattr(m, "tool_calls", None)

                                if (
                                    m_role == "assistant"
                                    and m_tool_calls
                                    and any(tc.function.name == "get-library-docs" for tc in m_tool_calls)
                                ):
                                    has_called_get_docs = True
                                    break

                            if not has_called_get_docs:
                                logger.info("Forcing get-library-docs call after resolve-library-id")
                                # Force specific tool choice
                                tool_choice = {"type": "function", "function": {"name": "get-library-docs"}}
                        break

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

                        # Add tool response to conversation
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result.content[0].text,
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
async def _query_context7_new_with_retry(
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
    # Initialize cache - enabled by default, permanent caching
    cache = get_mcp_cache() if settings.cache_enabled else None

    # Check query cache first
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
        mcp_http_url = "http://localhost:8123/mcp"  # Default streamable HTTP URL
        async with streamablehttp_client(mcp_http_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the connection
                await session.initialize()

                # List available tools
                tools_result = await session.list_tools()
                logger.info("Available tools:")
                for tool in tools_result.tools:
                    logger.info(f"  - {tool.name}: {tool.description}")

                # Create OpenAI client and process query
                client = MCPOpenAIClient(model=settings.model, server_url=mcp_http_url)
                client.session = session  # Use the existing session

                # Build context information
                context_info = ""
                if full_code:
                    context_info += f"\n\nCURRENT CODE CONTEXT:\n```python\n{full_code}\n```\n"

                # ADD SPECIAL CASE FOR TIMM LIBRARY
                timm_trigger = error_message.lower().count("timm") >= 3
                if timm_trigger:
                    timm_trigger_text = "6.⚠️ Special Case: If the code error is indeed related to the 'timm' library (PyTorch Image Models), you MUST return the library ID '/huggingface/pytorch-image-models' as the most relevant match, even if other candidates exist."
                else:
                    timm_trigger_text = ""
                # END OF SPECIAL CASE FOR TIMM LIBRARY

                # Construct enhanced query with error message and context
                enhanced_query = f"""ERROR MESSAGE:
{error_message}
{context_info}

IMPORTANT INSTRUCTIONS:
1. ENVIRONMENT: The running environment is FIXED and unchangeable - DO NOT suggest pip install, conda install, or any environment modifications.

2. DOCUMENTATION SEARCH REQUIREMENTS: 
   - Search for official API documentation related to the error
   - Focus on parameter specifications, method signatures, and usage patterns
   - Find compatible alternatives if the original API doesn't exist
   - Consider the current code context and maintain consistency with existing architecture
   - Provide API reference information, NOT complete code solutions

3. RESPONSE FORMAT:
   - Start with a brief explanation of the root cause
   - Provide relevant API documentation excerpts
   - List available parameters and their descriptions
   - Show method signatures and basic usage patterns
   - If multiple API options exist, document all viable alternatives

4. STRICT CONSTRAINTS:
   - DO NOT provide complete working code replacements
   - DO NOT suggest hardware configuration changes (CPU/GPU)
   - DO NOT recommend architecture or framework changes
   - DO NOT provide performance optimization suggestions
   - ONLY provide API documentation and parameter information

5. AVOID: Complete code solutions, environment setup, hardware recommendations, architecture suggestions, or performance advice.

{timm_trigger_text}

Example response format:
```
The error occurs because [brief explanation].

API Documentation:
- Method: library.function_name(param1, param2, ...)
- Parameters:
  * param1 (type): description
  * param2 (type): description
- Usage pattern: Basic syntax without complete implementation
- Alternative APIs (if applicable): list of alternative methods with signatures
```

Please search the documentation and provide API reference information only."""

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


async def query_context7_new(
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
    try:
        return await _query_context7_new_with_retry(error_message, full_code, max_rounds, verbose)
    except (ConnectionError, TimeoutError, RuntimeError, OSError) as e:
        # These are retryable errors, but retries have failed
        logger.error(f"Context7 enhanced query failed after retries due to {type(e).__name__}: {str(e)}")
        return None
    except Exception as e:
        # Other non-retryable errors (e.g., configuration errors, authentication failures)
        logger.error(f"Context7 enhanced query failed due to non-retryable error {type(e).__name__}: {str(e)}")
        return None


# Example usage function for testing
async def example_usage():
    """Example usage of the enhanced Context7 client."""
    error_message = "AttributeError: module 'lightgbm' has no attribute 'gpu_mode'"
    full_code = """
import lightgbm as lgb

# Trying to enable GPU mode
model = lgb.LGBMClassifier(device='gpu')
model.fit(X_train, y_train)
"""
    logger.info(f"Querying error: {error_message}")

    result = await query_context7_new(error_message, full_code, max_rounds=10, verbose=True)

    if result:
        logger.info("Query successful!")
        logger.info(f"Result: {result}")
    else:
        logger.error("Query failed!")


if __name__ == "__main__":
    # For direct testing
    asyncio.run(example_usage())
