"""MCP Unified Interface - Clean and Simple

This module provides a single, clean interface for executing MCP queries.
All legacy interfaces have been removed in favor of the new MCPAgent and mcp_execute APIs.
"""

import asyncio
import concurrent.futures
from functools import wraps
from typing import List, Optional, Union

from litellm import RateLimitError

from rdagent.components.mcp.conf import is_mcp_enabled
from rdagent.components.mcp.connector import MCPConnectionError
from rdagent.components.mcp.registry import get_global_registry
from rdagent.log import rdagent_logger as logger


def _mcp_enabled_check(func):
    """Decorator to ensure MCP is enabled before executing."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not is_mcp_enabled():
            logger.error("MCP system is globally disabled")
            return None

        try:
            registry = get_global_registry()
            await registry.ensure_initialized(verbose=kwargs.get("verbose", False))
            return await func(registry, *args, **kwargs)
        except (RateLimitError, MCPConnectionError) as e:
            logger.warning(f"MCP query encountered retryable error: {e}")
            raise
        except Exception as e:
            logger.error(f"MCP query failed with unexpected error: {e}")
            return None

    return wrapper


@_mcp_enabled_check
async def mcp_execute(
    registry, query: str, services: Optional[Union[str, List[str]]] = None, **kwargs
) -> Optional[str]:
    """
    Execute MCP query with specified services.

    This is the core execution function that replaces all legacy interfaces.
    It provides a clean, unified way to execute queries against MCP services.

    Args:
        query: The query/error message to process
        services: Optional service specification:
            - None: Use all available services (auto mode)
            - str: Use a specific service
            - List[str]: Use specified services
        **kwargs: Additional parameters passed to the handler

    Returns:
        Response from the MCP service(s), or None if failed

    Examples:
        # Auto mode - uses all available services
        result = await mcp_execute("ImportError: No module named 'sklearn'")

        # Single service mode
        result = await mcp_execute("error message", services="context7")

        # Multi-service mode
        result = await mcp_execute("error message", services=["context7", "deepwiki"])
    """
    return await registry.query(query, services=services, **kwargs)


def mcp_execute_sync(
    query: str, services: Optional[Union[str, List[str]]] = None, timeout: float = 180, **kwargs
) -> Optional[str]:
    """
    Synchronous version of mcp_execute for non-async contexts.

    This function runs MCP queries in a separate thread with proper cleanup
    to avoid event loop conflicts. It replaces execute_mcp_query_isolated.

    Args:
        query: The query content to send to MCP service
        services: MCP service(s) to use (None for all)
        timeout: Total timeout in seconds (default: 180)
        **kwargs: Additional keyword arguments to pass to mcp_execute

    Returns:
        Query result if successful, None otherwise

    Examples:
        # Auto mode
        result = mcp_execute_sync("ImportError: No module named 'sklearn'")

        # Specific service
        result = mcp_execute_sync("error message", services="context7")
    """

    def run_mcp_in_new_loop() -> Optional[str]:
        """Run MCP query in a new event loop to avoid conflicts."""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(mcp_execute(query, services=services, **kwargs))
        finally:
            # Graceful shutdown to avoid warnings
            pending = asyncio.all_tasks(new_loop)
            for task in pending:
                task.cancel()

            if pending:
                new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            new_loop.run_until_complete(new_loop.shutdown_asyncgens())
            new_loop.run_until_complete(asyncio.sleep(0.1))
            new_loop.close()

    # Execute in thread pool to avoid event loop conflicts
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_mcp_in_new_loop)
        return future.result(timeout=timeout)
