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
            logger.warning("MCP system is globally disabled")
            return None

        try:
            registry = get_global_registry()

            # Ensure services are initialized (only once)
            await registry.ensure_initialized()

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
        logger.info("Auto-registered all enabled MCP services from configuration")
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
async def query_mcp(service_name: str, query: str, **kwargs) -> Optional[str]:
    """Query a specific MCP service.

    Args:
        service_name: Name of the MCP service to query
        query: The query/error message to process
        **kwargs: Additional parameters passed to the handler

    Returns:
        Response from the MCP service, or None if failed
    """
    registry = kwargs.pop("_registry")  # Extract registry injected by decorator
    return await registry.query_service(service_name, query, **kwargs)


@mcp_api_handler
async def query_mcp_auto(query: str, **kwargs) -> Optional[str]:
    """Automatically select and query the best available MCP service.

    Args:
        query: The query/error message to process
        **kwargs: Additional parameters passed to the handler

    Returns:
        Response from the selected MCP service, or None if failed
    """
    registry = kwargs.pop("_registry")  # Extract registry injected by decorator
    return await registry.query_auto(query, **kwargs)


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
