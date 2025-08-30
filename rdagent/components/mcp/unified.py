from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Union

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


async def initialize_mcp_registry(config_path: Optional[Path] = None) -> MCPRegistry:
    """Initialize MCP registry with configuration file.

    Args:
        config_path: Path to MCP configuration file. If None, uses default path.

    Returns:
        Initialized MCP registry
    """
    if config_path is None:
        config_path = Path.cwd() / "mcp_config.json"

    registry = MCPRegistry.from_config_file(config_path)

    # Auto-register all enabled services from configuration
    try:
        await registry.auto_register_all_services()
        # Registry will log which services are registered
    except Exception as e:
        logger.error(f"Failed to auto-register MCP services: {e}")

    set_global_registry(registry)
    return registry


def register_mcp_handler(service_name: str, handler) -> bool:
    """Register a custom MCP handler.

    Args:
        service_name: Name of the MCP service
        handler: Handler instance implementing MCPHandler protocol

    Returns:
        True if registration successful, False otherwise
    """
    try:
        registry = get_global_registry()
        registry.register_handler(service_name, handler)
        return True
    except Exception as e:
        logger.error(f"Failed to register handler for '{service_name}': {e}")
        return False


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
    """Check if a specific MCP service is available."""
    try:
        registry = get_global_registry()
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
