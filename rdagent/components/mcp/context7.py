"""Context7 MCP integration for documentation search.

This module provides a simplified interface for querying documentation
using MCP (Model Context Protocol) tools.
"""

import asyncio
import time
from typing import Optional

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from llama_index.tools.mcp import aget_tools_from_mcp_url

from rdagent.components.mcp.cache import get_mcp_cache
from rdagent.components.mcp.util import get_context7_settings
from rdagent.log import rdagent_logger as logger


async def query_context7(
    error_message: str,
) -> Optional[str]:
    """Query context7 documentation for error resolution.

    Args:
        error_message: The error message or traceback to search for
    Returns:
        Documentation search result as string, or None if failed
    """
    try:
        # Load configuration using pydantic settings
        settings = get_context7_settings()
        print(settings)
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
        agent = ReActAgent(tools=tools, llm=llm, verbose=True)
        ctx = Context(agent)

        # Record time for agent execution
        agent_start_time = time.time()

        # Construct query with error message and context7 instruction
        # TODO: how to fix the agent to force the two tools to be used
        # TODO: how to extend to more apis (currently only gpt models through llama_index)

        query = f"""{error_message}

IMPORTANT INSTRUCTIONS:
1. ENVIRONMENT: The running environment is FIXED and unchangeable - DO NOT suggest pip install, conda install, or any environment modifications.

2. SOLUTION REQUIREMENTS: 
   - Provide WORKING CODE that directly replaces the problematic code
   - Include specific import statements if needed
   - Show the exact API usage from the official documentation
   - If the original API doesn't exist, provide the correct alternative API with identical functionality

3. RESPONSE FORMAT:
   - Start with a brief explanation of the root cause
   - Provide the corrected code block that can be directly copy-pasted
   - Include any necessary import changes
   - If multiple solutions exist, show the most straightforward one first

4. DOCUMENTATION SEARCH: Use context7 to find the official API documentation and provide solutions based on the actual available methods and parameters.

5. AVOID: Version upgrade suggestions, environment setup, debugging commands, or theoretical explanations without concrete code solutions.

Example response format:
```
The error occurs because [brief explanation].

Solution - Replace the problematic code with:
```python
# Corrected imports (if needed)
import library as lib

# Working replacement code
corrected_code_here
```
```

Please search the documentation and provide a practical, copy-paste solution."""

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

    except Exception as e:
        logger.error(f"Context7 query failed: {str(e)}")
        return None


async def main():
    """Main function for testing context7 functionality."""
    error_msg = """### TRACEBACK: Traceback (most recent call last):\nFile \"/workspace/RD-Agent/git_ignore_folder/RD-Agent_workspace/55141c6414284b9f8512f998b4b91043/main.py\", line 540, in <module>\nmain()\nFile \"/workspace/RD-Agent/git_ignore_folder/RD-Agent_workspace/55141c6414284b9f8/512f998b4b91043/main.py\", line 440, in main\ntrain_transforms = build_transforms('train')\n^^^^^^^^^^^^^^^^^^^^^^^^^\nFile \"/workspace/RD-Agent/git_ignore_folder/RD-Agent_workspace/55141c6414284b9f8512f998b4b91043/main.py\", line 149, in build_transforms\nA.RandAugment(n=2, m=9),\n^^^^^^^^^^^^^\nAttributeError: module 'albumentations' has no attribute 'RandAugment'\n### SUPPLEMENTARY_INFO: import albumentations as A\nfrom albumentations.pytorch import ToTensorV2"""

    result = await query_context7(error_msg)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
