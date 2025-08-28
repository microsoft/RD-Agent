"""MCP Service Registry

This module provides configuration-driven service discovery and management
for multiple MCP services. It supports dynamic registration and routing.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from rdagent.components.mcp.connector import (
    StreamableHTTPConfig,
    StreamableHTTPConnector,
    create_mcp_session_with_retry,
)
from rdagent.log import rdagent_logger as logger


class MCPServiceConfig(BaseModel):
    """Configuration for a single MCP service."""

    name: str = Field(..., description="Service name")
    url: str = Field(..., description="Service URL")
    timeout: float = Field(default=30.0, description="Connection timeout")
    headers: Optional[Dict[str, Any]] = Field(default=None, description="HTTP headers")
    handler: str = Field(..., description="Handler class name")
    enabled: bool = Field(default=True, description="Whether service is enabled")

    # Additional service-specific configuration
    extra_config: Dict[str, Any] = Field(default_factory=dict, description="Extra configuration")

    def to_connector_config(self) -> StreamableHTTPConfig:
        """Convert to StreamableHTTPConfig."""
        return StreamableHTTPConfig(url=self.url, timeout=self.timeout, headers=self.headers or {})


class MCPRegistryConfig(BaseModel):
    """Configuration for the entire MCP registry."""

    mcp_services: Dict[str, MCPServiceConfig] = Field(default_factory=dict, description="Map of service name to config")
    default_timeout: float = Field(default=120.0, description="Default timeout for services")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPRegistryConfig":
        """Create from dictionary, handling nested service configs."""
        services = {}

        # Handle both flat and nested config formats
        service_data = data.get("mcpServices", data.get("mcp_services", {}))

        for name, config in service_data.items():
            if isinstance(config, dict):
                # Add name to config if not present
                config["name"] = name
                # Add default handler if not specified
                if "handler" not in config:
                    config["handler"] = f"{name.title()}Handler"
                services[name] = MCPServiceConfig(**config)
            else:
                logger.warning(f"Invalid configuration for service '{name}': {config}")

        return cls(mcp_services=services)

    @classmethod
    def from_file(cls, config_path: Path) -> "MCPRegistryConfig":
        """Load configuration from JSON file."""
        if not config_path.exists():
            logger.warning(f"MCP config file not found: {config_path}")
            return cls()

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to load MCP config from {config_path}: {e}")
            return cls()
        except Exception as e:
            logger.error(f"Unexpected error loading MCP config: {e}")
            return cls()


@runtime_checkable
class MCPHandler(Protocol):
    """Protocol for MCP service handlers."""

    async def process_query(self, connector: StreamableHTTPConnector, query: str, **kwargs) -> str:
        """Process a query using the given connector."""
        ...

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about this service."""
        ...


class MCPRegistry:
    """Registry for managing multiple MCP services."""

    def __init__(self, config: Optional[MCPRegistryConfig] = None):
        self.config = config or MCPRegistryConfig()
        self._handlers: Dict[str, MCPHandler] = {}

    @classmethod
    def from_config_file(cls, config_path: Path) -> "MCPRegistry":
        """Create registry from configuration file."""
        config = MCPRegistryConfig.from_file(config_path)
        return cls(config)

    def register_handler(self, service_name: str, handler: MCPHandler):
        """Register a handler for a service."""
        if service_name not in self.config.mcp_services:
            raise ValueError(f"Service '{service_name}' not configured")

        self._handlers[service_name] = handler
        logger.info(f"Registered handler for service '{service_name}'")

    def get_service_config(self, service_name: str) -> Optional[MCPServiceConfig]:
        """Get configuration for a service."""
        return self.config.mcp_services.get(service_name)

    def get_enabled_services(self) -> List[str]:
        """Get list of enabled service names."""
        return [name for name, config in self.config.mcp_services.items() if config.enabled]

    def has_service(self, service_name: str) -> bool:
        """Check if service is configured and enabled."""
        config = self.get_service_config(service_name)
        return config is not None and config.enabled

    def has_handler(self, service_name: str) -> bool:
        """Check if handler is registered for service."""
        return service_name in self._handlers

    def _import_handler_class(self, handler_class_name: str):
        """动态导入Handler类"""
        if handler_class_name == "Context7Handler":
            from rdagent.components.mcp.context7.handler import Context7Handler

            return Context7Handler
        else:
            # 未来可以支持更多Handler类型
            # 可以通过完整模块路径动态导入: "module.path:ClassName"
            raise ValueError(f"Unknown handler class: {handler_class_name}")

    async def auto_register_all_services(self):
        """基于mcp_config.json自动注册所有启用的服务"""
        for name, config in self.config.mcp_services.items():
            if config.enabled and not self.has_handler(name):
                try:
                    # 动态导入Handler类
                    handler_class = self._import_handler_class(config.handler)

                    # 创建Handler实例，传入服务配置
                    handler = handler_class(name, service_url=config.url, **config.extra_config)

                    # 注册Handler
                    self._handlers[name] = handler
                    logger.info(f"Auto-registered handler for service '{name}' with class '{config.handler}'")

                except Exception as e:
                    logger.error(f"Failed to auto-register service '{name}': {e}")

    async def query_parallel(self, services: List[str], query: str, **kwargs) -> Dict[str, str]:
        """并发查询多个服务，返回所有结果"""
        if not services:
            logger.warning("No services specified for parallel query")
            return {}

        # 过滤出有效的服务
        valid_services = [service for service in services if self.has_service(service) and self.has_handler(service)]

        if not valid_services:
            logger.warning("No valid services available for parallel query")
            return {}

        logger.info(f"Starting parallel query for services: {valid_services}")

        # 创建并发任务
        tasks = []
        for service in valid_services:
            task = asyncio.create_task(self.query_service(service, query, **kwargs), name=f"query_{service}")
            tasks.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果和异常
        service_results = {}
        for service, result in zip(valid_services, results):
            if isinstance(result, Exception):
                service_results[service] = f"Error: {str(result)}"
                logger.error(f"Service '{service}' failed: {result}")
            elif result is not None:
                service_results[service] = result
                logger.info(f"Service '{service}' completed successfully")
            else:
                service_results[service] = "No result returned"
                logger.warning(f"Service '{service}' returned no result")

        return service_results

    async def query_service(self, service_name: str, query: str, **kwargs) -> Optional[str]:
        """Query a specific service."""
        # Check if service exists and is enabled
        if not self.has_service(service_name):
            logger.warning(f"Service '{service_name}' not available")
            return None

        # Check if handler is registered
        if not self.has_handler(service_name):
            logger.warning(f"No handler registered for service '{service_name}'")
            return None

        # Get service configuration and handler
        config = self.get_service_config(service_name)
        handler = self._handlers[service_name]

        # Create connector with retry mechanism
        connector_config = config.to_connector_config()

        try:
            # Use the retry mechanism for creating connector
            connector = await create_mcp_session_with_retry(connector_config)

            # Process query using handler
            result = await handler.process_query(connector, query, **kwargs)
            logger.info(f"Query processed successfully by service '{service_name}'")
            return result
        except Exception as e:
            logger.error(f"Error querying service '{service_name}': {e}")
            return None

    async def query_auto(self, query: str, **kwargs) -> Optional[str]:
        """Automatically select and query the best available service."""
        enabled_services = self.get_enabled_services()

        if not enabled_services:
            logger.warning("No enabled MCP services available")
            return None

        # For now, try the first available service
        # TODO: Implement smarter service selection logic
        service_name = enabled_services[0]
        logger.info(f"🔍 Auto-selecting MCP service: {service_name}", tag="mcp_query")

        return await self.query_service(service_name, query, **kwargs)

    def get_service_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered services."""
        info = {}

        for name, config in self.config.mcp_services.items():
            service_info = {
                "enabled": config.enabled,
                "url": config.url,
                "handler": config.handler,
                "has_handler_registered": self.has_handler(name),
            }

            # Add handler-specific info if available
            if self.has_handler(name):
                try:
                    handler_info = self._handlers[name].get_service_info()
                    service_info.update(handler_info)
                except Exception as e:
                    logger.warning(f"Failed to get info from handler '{name}': {e}")

            info[name] = service_info

        return info


# Global registry instance
_global_registry: Optional[MCPRegistry] = None


def get_global_registry() -> MCPRegistry:
    """Get or create the global MCP registry."""
    global _global_registry

    if _global_registry is None:
        # Try to load from default config path
        config_path = Path.cwd() / "mcp_config.json"
        _global_registry = MCPRegistry.from_config_file(config_path)

    return _global_registry


def set_global_registry(registry: MCPRegistry):
    """Set the global MCP registry."""
    global _global_registry
    _global_registry = registry
