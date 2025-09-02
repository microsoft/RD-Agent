import asyncio
import concurrent.futures
from functools import wraps
from pathlib import Path
from typing import Any, List, Optional, Union

from rdagent.components.mcp.conf import is_mcp_enabled
from rdagent.components.mcp.registry import (
    MCPRegistry,
    get_global_registry,
    set_global_registry,
)
from rdagent.log import rdagent_logger as logger


def mcp_api_handler(func):
    """Decorator to handle common MCP API patterns: enabled check, registry init, error handling."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not is_mcp_enabled():
            logger.error("MCP system is globally disabled")
            return None

        try:
            registry = get_global_registry()

            # Extract verbose flag for initialization details
            verbose = kwargs.get("verbose", False)

            # Ensure services are initialized (only once)
            await registry.ensure_initialized(verbose=verbose)

            # Inject registry into kwargs for the function
            kwargs["_registry"] = registry
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            return None

    return wrapper


@mcp_api_handler
async def query_mcp(query: str, services: Optional[Union[str, List[str]]] = None, **kwargs) -> Optional[str]:
    """Unified MCP query interface supporting flexible service selection.

    This function provides three usage modes:
    1. Auto mode (default): Uses all available services in parallel
    2. Single service mode: Query a specific service directly
    3. Multi-service mode: Use specified services in parallel

    Args:
        query: The query/error message to process
        services: Optional service specification:
            - None: Use all available services (auto mode)
            - str: Use a specific service
            - List[str]: Use specified services in parallel
        **kwargs: Additional parameters passed to the handler

    Returns:
        Response from the MCP service(s), or None if failed

    Examples:
        # Auto mode - uses all available services
        await query_mcp("error message")

        # Single service mode
        await query_mcp("error message", services="context7")

        # Multi-service mode - uses specified services in parallel
        await query_mcp("error message", services=["context7", "simple_code_search"])
    """
    registry = kwargs.pop("_registry")  # Extract registry injected by decorator

    # Check verbose flag and show service status for debugging
    verbose = kwargs.get("verbose", False)
    if verbose:
        status = get_service_status()
        if status["available_services"]:
            logger.info(f"🔍 MCP services available: {status['available_services']}", tag="mcp_status")
        else:
            logger.error("⚠️ No MCP services available for query", tag="mcp_status")

    # Handle different service specifications
    if isinstance(services, str):
        # Single service mode: direct query
        return await registry.query_service(services, query, **kwargs)
    else:
        # Multi-service mode (including None for auto mode)
        # query_auto handles both None (all services) and list cases
        return await registry.query_auto(query, services=services, **kwargs)


# Utility functions
def list_available_mcp_services() -> list:
    """List all available MCP service names."""
    try:
        registry = get_global_registry()
        return registry.get_enabled_services()
    except Exception as e:
        logger.error(f"Failed to list MCP services: {e}")
        return []


def is_service_available(service_name: str) -> bool:
    """Check if a specific MCP service is available.

    Note: This is a synchronous function that checks current registry state.
    Services are initialized asynchronously on first query_mcp call.
    """
    try:
        registry = get_global_registry()

        # Try to trigger initialization if not already done
        # This is a best-effort attempt for synchronous context
        if not registry._initialized and not registry._handlers:
            # Service is configured but handler not yet registered
            # Return True if service is enabled, as it will be initialized on first use
            return registry.has_service(service_name)

        return registry.has_service(service_name) and registry.has_handler(service_name)
    except Exception:
        return False


def get_service_status() -> dict:
    """Get status of all MCP services."""
    try:
        available_services = list_available_mcp_services()
        status = {"mcp_enabled": is_mcp_enabled(), "available_services": available_services, "service_details": {}}

        # Check each service status
        for service_name in available_services:
            status["service_details"][service_name] = {
                "available": is_service_available(service_name),
                "enabled": True,  # If it's in available_services, it's enabled
            }

        return status
    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        return {"mcp_enabled": is_mcp_enabled(), "available_services": [], "service_details": {}}


def execute_mcp_query_isolated(
    query: str,
    services: str | List[str],
    timeout: float = 180,
    verbose: bool = True,
    **mcp_kwargs: Any,
) -> Optional[str]:
    """
    Execute MCP query in an isolated event loop with proper cleanup.

    This function runs MCP queries in a separate thread with a new event loop
    to avoid conflicts with existing async contexts. It includes comprehensive
    resource cleanup to prevent "Task was destroyed but it is pending" warnings.

    Args:
        query: The query content to send to MCP service
        services: MCP service(s) to use (e.g., "context7", ["context7", "deepwiki"])
        timeout: Total timeout in seconds for the operation (default: 180)
        verbose: Enable verbose logging for debugging (default: True)
        **mcp_kwargs: Additional keyword arguments to pass to query_mcp

    Returns:
        Optional[str]: Query result if successful, None otherwise

    Raises:
        concurrent.futures.TimeoutError: If the query exceeds the specified timeout
        Exception: For other MCP-related errors
    """

    def run_mcp_sync() -> Optional[str]:
        """Run MCP query in a new event loop to avoid conflicts"""
        # Create new event loop to avoid conflicts with existing loop
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(
                query_mcp(
                    query,
                    services=services,
                    verbose=verbose,
                    **mcp_kwargs,
                )
            )
        finally:
            # Graceful shutdown to avoid "Task was destroyed but it is pending" warnings
            # 1. Cancel all pending tasks
            pending = asyncio.all_tasks(new_loop)
            for task in pending:
                task.cancel()

            # 2. Wait for cancellation to complete
            if pending:
                new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            # 3. Shutdown async generators
            new_loop.run_until_complete(new_loop.shutdown_asyncgens())

            # 4. Give SSE streams time to clean up
            new_loop.run_until_complete(asyncio.sleep(0.1))

            # 5. Close the event loop
            new_loop.close()

    # Execute in thread pool to avoid event loop conflicts
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_mcp_sync)
        return future.result(timeout=timeout)
