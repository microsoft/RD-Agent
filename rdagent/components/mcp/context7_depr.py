"""Context7 MCP integration for documentation search.

This module provides a simplified interface for querying documentation
using MCP (Model Context Protocol) tools.

Packages needed:
# llama-index
# llama-index-tools-mcp
"""

import asyncio
import time
from typing import Optional

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from llama_index.tools.mcp import aget_tools_from_mcp_url
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rdagent.components.mcp.cache import get_mcp_cache
from rdagent.components.mcp.util import get_context7_settings
from rdagent.log import rdagent_logger as logger


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
    error_message: str, full_code: Optional[str] = None, verbose: bool = False
) -> Optional[str]:
    """Internal function with retry mechanism for Context7 queries.

    Args:
        error_message: The error message or traceback to search for
        full_code: Complete code context for better understanding (optional)
        verbose: Enable verbose logging for ReAct agent (default: False)
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

    # Try to get tools from cache first
    tools = cache.get_tools(settings.mcp_url) if cache else None

    if tools is None:
        # Cache miss or cache disabled, load tools from URL
        tool_start_time = time.time()
        tools = await aget_tools_from_mcp_url(settings.mcp_url)
        tool_end_time = time.time()
        logger.info(f"Context7 tool loading time: {tool_end_time - tool_start_time:.2f} seconds")

        # Cache the tools for future use
        if cache:
            cache.set_tools(settings.mcp_url, tools)
    else:
        logger.info("Using cached tools, loading time: 0.00 seconds")

    # Initialize LLM with OpenAI configuration
    llm = OpenAI(model=settings.model, api_key=settings.api_key, api_base=settings.api_base)

    # Create ReAct agent with loaded tools
    agent = ReActAgent(tools=tools, llm=llm, verbose=verbose)
    ctx = Context(agent)

    # Record time for agent execution
    agent_start_time = time.time()

    # Construct query with error message and context7 instruction
    # TODO: how to fix the agent to force the two tools to be used
    # TODO: how to extend to more apis (currently only gpt models through llama_index)

    # ADD SPECIAL CASE FOR TIMM LIBRARY
    timm_trigger = error_message.lower().count("timm") >= 3
    if timm_trigger:
        timm_trigger_text = "6.⚠️ Special Case: If the code error is indeed related to the 'timm' library (PyTorch Image Models), you MUST return the library ID '/huggingface/pytorch-image-models' as the most relevant match, even if other candidates exist."
    else:
        timm_trigger_text = ""
    # END OF SPECIAL CASE FOR TIMM LIBRARY

    # Build context information
    context_info = ""
    if full_code:
        context_info += f"\n\nCURRENT CODE CONTEXT:\n```python\n{full_code}\n```\n"

    query = f"""ERROR MESSAGE:
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

    # Execute agent query
    response = await agent.run(query, ctx=ctx)

    agent_end_time = time.time()
    logger.info(f"Context7 agent execution time: {agent_end_time - agent_start_time:.2f} seconds")

    # Calculate and display total execution time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Context7 total execution time: {total_time:.2f} seconds")
    logger.info(f"Context7 execution completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

    result = str(response)

    # Cache the query result
    if cache:
        cache.set_query_result(error_message, result)
        cache.log_cache_stats()

    return result


async def query_context7(error_message: str, full_code: Optional[str] = None, verbose: bool = False) -> Optional[str]:
    """Query context7 documentation for error resolution with retry mechanism.

    Args:
        error_message: The error message or traceback to search for
        full_code: Complete code context for better understanding (optional)
        verbose: Enable verbose logging for ReAct agent (default: False)
    Returns:
        Documentation search result as string, or None if failed
    """
    try:
        return await _query_context7_with_retry(error_message, full_code, verbose)
    except (ConnectionError, TimeoutError, RuntimeError, OSError) as e:
        # These are retryable errors, but retries have failed
        logger.error(f"Context7 query failed after retries due to {type(e).__name__}: {str(e)}")
        return None
    except Exception as e:
        # Other non-retryable errors (e.g., configuration errors, authentication failures)
        logger.error(f"Context7 query failed due to non-retryable error {type(e).__name__}: {str(e)}")
        return None
