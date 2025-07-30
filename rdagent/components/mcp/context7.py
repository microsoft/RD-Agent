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

from rdagent.components.mcp.util import load_mcp_config
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
        # Load configuration using existing LLM settings
        config = load_mcp_config()
        mcp_url = config.get("mcp_url")
        openai_model = config.get("model")
        openai_api_key = config.get("api_key")
        openai_api_base = config.get("api_base")

        # Record start time for execution timing
        start_time = time.time()
        tool_start_time = time.time()
        tools = await aget_tools_from_mcp_url(mcp_url)
        tool_end_time = time.time()
        logger.info(f"Context7 tool loading time: {tool_end_time - tool_start_time:.2f} seconds")

        # Initialize LLM with OpenAI configuration
        llm = OpenAI(model=openai_model, api_key=openai_api_key, api_base=openai_api_base)

        # Create ReAct agent with loaded tools
        agent = ReActAgent(tools=tools, llm=llm, verbose=True)
        ctx = Context(agent)

        # Record time for agent execution
        agent_start_time = time.time()

        # Construct query with error message and context7 instruction
        # TODO: how to fix the agent to force the two tools to be used
        # TODO: how to extend to more apis (currently only gpt models through llama_index)
        query = f"""
To solve this error: {error_message}

Please follow these exact steps:
1. First use resolve-library-id to find the library ID for albumentations
2. Then use get-library-docs to get the documentation for albumentations  
3. Then use resolve-library-id to find the library ID for albucore
4. Then use get-library-docs to get the documentation for albucore
5. Provide a solution based on both documentations

You MUST use both resolve-library-id and get-library-docs tools for each library.
"""

        # Execute agent query
        response = await agent.run(query, ctx=ctx)

        agent_end_time = time.time()
        logger.info(f"Context7 agent execution time: {agent_end_time - agent_start_time:.2f} seconds")

        # Calculate and display total execution time
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Context7 total execution time: {total_time:.2f} seconds")
        logger.info(f"Context7 execution completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

        return str(response)

    except Exception as e:
        logger.error(f"Context7 query failed: {str(e)}")
        return None


async def main():
    """Main function for testing context7 functionality."""
    error_msg = """Traceback (most recent call last):
File "/workspace/RD-Agent/git_ignore_folder/RD-Agent_workspace/7f247f1f37894dd1a1e3057552e1972d/main.py", line 13, in <module>
import albumentations as A
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/albumentations/__init__.py", line 6, in <module>
from .augmentations import *
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/albumentations/augmentations/__init__.py", line 1, in <module>
from .blur.functional import *
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/albumentations/augmentations/blur/__init__.py", line 1, in <module>
from .functional import *
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/albumentations/augmentations/blur/functional.py", line 9, in <module>
from albucore.utils import clipped, maybe_process_in_chunks, preserve_channel_dim
ImportError: cannot import name 'preserve_channel_dim' from 'albucore.utils' (/opt/conda/envs/kaggle/lib/python3.11/site-packages/albucore/utils.py)
This is likely due to (3) environment/dependency problems and not an algorithmic or syntax issue."""
    result = await query_context7(error_msg)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
