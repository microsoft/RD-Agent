"""Context7 MCP integration for documentation search.

This module provides a simplified interface for querying documentation
using MCP (Model Context Protocol) tools.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Optional

import yaml
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from llama_index.tools.mcp import aget_tools_from_mcp_url

from rdagent.log import rdagent_logger as logger


def load_config():
    """Load configuration from config.yaml file.

    Returns:
        dict: Configuration dictionary with default values if file not found
    """
    try:
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config.yaml: {e}.")


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
        # Load configuration from config.yaml
        config = load_config()

        # Set values with priority: parameter > environment variable > config file
        mcp_url = config.get("mcp_url")
        openai_model = config.get("openai_model")
        openai_api_key = config.get("openai_api_key")
        openai_api_base = config.get("openai_api_base")

        # Record start time for execution timing
        start_time = time.time()

        logger.info(f"Context7 execution started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

        # Record time for tool loading
        tool_start_time = time.time()
        tools = await aget_tools_from_mcp_url(mcp_url)
        tool_end_time = time.time()
        logger.info(f"Context7 tool loading time: {tool_end_time - tool_start_time:.2f} seconds")

        # Initialize LLM with OpenAI configuration
        llm = OpenAI(model=openai_model, api_key=openai_api_key, api_base=openai_api_base)

        # Create ReAct agent with loaded tools
        agent = ReActAgent(tools=tools, llm=llm)
        ctx = Context(agent)

        # Record time for agent execution
        agent_start_time = time.time()

        # Construct query with error message and context7 instruction
        query = f"{error_message} use context7"

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
