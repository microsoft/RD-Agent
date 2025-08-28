"""Unified MCP Query Interface

This module provides the unified interface for querying MCP services with support
for auto service selection and parallel querying.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from rdagent.components.mcp.conf import is_mcp_enabled
from rdagent.components.mcp.context7.handler import Context7Handler
from rdagent.components.mcp.registry import (
    MCPRegistry,
    get_global_registry,
    set_global_registry,
)
from rdagent.log import rdagent_logger as logger


def _ensure_registry_initialized() -> MCPRegistry:
    """Ensure the global registry is initialized with default configuration."""
    registry = get_global_registry()

    # Check if Context7 handler is already registered
    if not registry.has_handler("context7"):
        try:
            # Auto-register Context7 handler with config if available
            service_config = registry.get_service_config("context7")
            config_dict = {
                "service_url": service_config.url if service_config else "http://localhost:8123/mcp",
                "extra_config": service_config.extra_config if service_config else {},
            }
            context7_handler = Context7Handler("context7", **config_dict)
            registry.register_handler("context7", context7_handler)
            logger.info("Auto-registered Context7 handler with available config")
        except Exception as e:
            logger.warning(f"Failed to auto-register Context7 handler: {e}")

    return registry


def initialize_mcp_registry(config_path: Optional[Path] = None) -> MCPRegistry:
    """Initialize MCP registry with configuration file.

    Args:
        config_path: Path to MCP configuration file. If None, uses default path.

    Returns:
        Initialized MCP registry
    """
    if config_path is None:
        config_path = Path.cwd() / "mcp_config.json"

    registry = MCPRegistry.from_config_file(config_path)

    # Auto-register Context7 handler if enabled
    if registry.has_service("context7"):
        try:
            # 获取Context7服务配置，包括extra_config
            service_config = registry.get_service_config("context7")
            config_dict = {
                "service_url": service_config.url if service_config else "http://localhost:8123/mcp",
                "extra_config": service_config.extra_config if service_config else {},
            }
            context7_handler = Context7Handler("context7", **config_dict)
            registry.register_handler("context7", context7_handler)
            logger.info("Registered Context7 handler with config from registry")
        except Exception as e:
            logger.error(f"Failed to register Context7 handler: {e}")

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


# 同步包装器函数（为兼容现有代码）
def is_service_available_sync(service_name: str) -> bool:
    """同步版本的服务可用性检查 - 为兼容现有代码"""
    import asyncio

    try:
        # 检查是否有运行中的事件循环
        asyncio.get_running_loop()
        # 如果有，创建任务但不等待（返回保守结果）
        return True  # 保守返回True，让实际查询时处理
    except RuntimeError:
        # 没有运行中的事件循环，可以安全使用 run
        try:
            return asyncio.run(is_service_available(service_name))
        except Exception:
            return False


def list_available_mcp_services_sync() -> list:
    """同步版本的服务列表 - 为兼容现有代码"""
    import asyncio

    try:
        asyncio.get_running_loop()
        return ["context7"]  # 返回默认服务列表
    except RuntimeError:
        try:
            return asyncio.run(list_available_mcp_services())
        except Exception:
            return []


# 辅助函数
def get_mcp_service_info() -> dict:
    """Get information about all registered MCP services."""
    try:
        registry = _ensure_registry_initialized()
        return registry.get_service_info()
    except Exception as e:
        logger.error(f"Failed to get MCP service info: {e}")
        return {}


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
