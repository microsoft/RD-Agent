"""Unified MCP Query Interface

This module provides the unified interface for querying MCP services with support
for auto service selection and parallel querying.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from rdagent.components.mcp.conf import is_mcp_enabled
from rdagent.components.mcp.registry import (
    MCPRegistry,
    get_global_registry,
    set_global_registry,
)
from rdagent.log import rdagent_logger as logger


def _ensure_registry_initialized() -> MCPRegistry:
    """Ensure the global registry is initialized and handlers are registered."""
    registry = get_global_registry()

    # Check if any services need registration
    enabled_services = registry.get_enabled_services()
    services_needing_registration = [name for name in enabled_services if not registry.has_handler(name)]

    if not services_needing_registration:
        return registry

    # Synchronously register handlers for immediate availability
    for service_name in services_needing_registration:
        try:
            service_config = registry.get_service_config(service_name)
            if not service_config:
                continue

            # Import handler class
            handler_class = registry._import_handler_class(service_config.handler)

            # Create handler instance
            handler = handler_class(service_name, service_url=service_config.url, **service_config.extra_config)

            # Register handler
            registry.register_handler(service_name, handler)
            logger.info(f"Sync-registered handler for service '{service_name}'")

        except Exception as e:
            logger.warning(f"Failed to sync-register service '{service_name}': {e}")

    return registry


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
        registry = _ensure_registry_initialized()
        registry.register_handler(service_name, handler)
        return True
    except Exception as e:
        logger.error(f"Failed to register handler for '{service_name}': {e}")
        return False


async def query_mcp(service_name: str, query: str, **kwargs) -> Optional[str]:
    """Query a specific MCP service.

    Args:
        service_name: Name of the MCP service to query
        query: The query/error message to process
        **kwargs: Additional parameters passed to the handler

    Returns:
        Response from the MCP service, or None if failed
    """
    if not is_mcp_enabled():
        logger.warning("MCP system is globally disabled")
        return None

    try:
        registry = _ensure_registry_initialized()
        result = await registry.query_service(service_name, query, **kwargs)
        return result
    except Exception as e:
        logger.error(f"MCP query failed for service '{service_name}': {e}")
        return None


async def query_mcp_auto(query: str, **kwargs) -> Optional[str]:
    """Automatically select and query the best available MCP service.

    Args:
        query: The query/error message to process
        **kwargs: Additional parameters passed to the handler

    Returns:
        Response from the selected MCP service, or None if failed
    """
    if not is_mcp_enabled():
        logger.warning("MCP system is globally disabled")
        return None

    try:
        registry = _ensure_registry_initialized()
        result = await registry.query_auto(query, **kwargs)
        return result
    except Exception as e:
        logger.error(f"Auto MCP query failed: {e}")
        return None


# Utility functions
def list_available_mcp_services() -> list:
    """List all available MCP service names."""
    try:
        registry = _ensure_registry_initialized()
        return registry.get_enabled_services()
    except Exception as e:
        logger.error(f"Failed to list MCP services: {e}")
        return []


def is_service_available(service_name: str) -> bool:
    """Check if a specific MCP service is available."""
    try:
        registry = _ensure_registry_initialized()
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
