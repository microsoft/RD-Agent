import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from rdagent.components.mcp.connector import (
    StreamableHTTPConfig,
    StreamableHTTPConnector,
)
from rdagent.log import rdagent_logger as logger
from rdagent.utils import get_module_by_module_path


class MCPServiceConfig(BaseModel):
    """Configuration for a single MCP service."""

    name: str = Field(..., description="Service name")
    url: str = Field(..., description="Service URL")
    timeout: float = Field(default=120.0, description="Connection timeout")
    headers: Optional[Dict[str, Any]] = Field(default=None, description="HTTP headers")
    handler: str = Field(
        ...,
        description="Handler specification: either legacy class name (e.g., 'Context7Handler') "
        "or full module path (e.g., 'module.path:ClassName')",
    )
    enabled: bool = Field(default=True, description="Whether service is enabled")

    # Additional service-specific configuration
    extra_config: Dict[str, Any] = Field(default_factory=dict, description="Extra configuration")

    def to_connector_config(self) -> StreamableHTTPConfig:
        """Convert to StreamableHTTPConfig."""
        return StreamableHTTPConfig(url=self.url, timeout=self.timeout, headers=self.headers or {})


class MCPRegistryConfig(BaseModel):
    """Configuration for the entire MCP registry."""

    mcp_services: Dict[str, MCPServiceConfig] = Field(default_factory=dict, description="Map of service name to config")

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


class MCPRegistry:
    """Registry for managing multiple MCP services."""

    def __init__(self, config: Optional[MCPRegistryConfig] = None):
        self.config = config or MCPRegistryConfig()
        self._handlers: Dict[str, Any] = {}
        self._initialized: bool = False

    @classmethod
    def from_config_file(cls, config_path: Path) -> "MCPRegistry":
        """Create registry from configuration file."""
        config = MCPRegistryConfig.from_file(config_path)
        return cls(config)

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

    def _import_handler_class(self, handler_spec: str, verbose: bool = False):
        """Dynamically import Handler class using full module path format only.

        Args:
            handler_spec: Handler specification in format 'module.path:ClassName'
            verbose: Whether to log import details

        Returns:
            Handler class object

        Raises:
            ValueError: If unable to import the specified Handler class
        """
        # Check for full module path format
        if ":" not in handler_spec:
            raise ValueError(
                f"Invalid handler specification: '{handler_spec}'. " "Must use format 'module.path:ClassName'."
            )

        try:
            # Parse module path and class name
            module_path, class_name = handler_spec.split(":", 1)

            # Dynamically import module
            module = get_module_by_module_path(module_path)

            # Get Handler class
            handler_class = getattr(module, class_name)

            # Log import success only in verbose mode
            if verbose:
                logger.info(f"Successfully imported handler: {handler_spec}")

            return handler_class

        except (ModuleNotFoundError, ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import handler '{handler_spec}': {e}") from e

    async def ensure_initialized(self, verbose: bool = False):
        """Ensure all services are initialized (only executed once)."""
        if not self._initialized:
            await self.auto_register_all_services(verbose=verbose)
            self._initialized = True
            # Always show which services were actually registered (important system status)
            registered_services = [name for name in self._handlers.keys()]
            if registered_services:
                logger.info(f"MCP registry initialized with services: {registered_services}")
            else:
                logger.info("MCP registry initialized (no services registered)")

    async def auto_register_all_services(self, verbose: bool = False):
        """Auto-register all enabled services from configuration."""
        for name, config in self.config.mcp_services.items():
            if config.enabled and not self.has_handler(name):
                try:
                    # Dynamically import Handler class
                    handler_class = self._import_handler_class(config.handler, verbose=verbose)

                    # Create Handler instance with service configuration
                    handler = handler_class(name, service_url=config.url, extra_config=config.extra_config)

                    # Register Handler
                    self._handlers[name] = handler
                    if verbose:
                        logger.info(f"Auto-registered handler for service '{name}' with class '{config.handler}'")

                except Exception as e:
                    logger.error(f"Failed to auto-register service '{name}': {e}")

    async def query(self, query: str, services: Optional[Union[str, List[str]]] = None, **kwargs) -> Optional[str]:
        """
        Unified query method for any number of services.

        This method treats single-service as a special case of multi-service,
        eliminating code duplication between query_service and query_auto.

        Args:
            query: The query to process
            services: Optional service specification:
                - None: Use all enabled services
                - str: Use a specific service
                - List[str]: Use specified services
            **kwargs: Additional parameters passed to the handler

        Returns:
            Response from the service(s), or None if failed
        """
        # Normalize services to a list
        if services is None:
            target_services = self.get_enabled_services()
        elif isinstance(services, str):
            target_services = [services]
        else:
            target_services = services

        if not target_services:
            logger.warning("No services specified or available")
            return None

        # Prepare connectors for all target services
        connectors = {}
        handler = None

        for service_name in target_services:
            # Check if service is configured and enabled
            if not self.has_service(service_name):
                logger.warning(f"Service '{service_name}' not available, skipping")
                continue

            # Check if handler is registered
            if not self.has_handler(service_name):
                logger.warning(f"No handler registered for service '{service_name}', skipping")
                continue

            # Get service configuration
            config = self.get_service_config(service_name)
            connector_config = config.to_connector_config()

            # Get handler (use any handler, they all have the same unified processing)
            if handler is None:
                handler = self._handlers[service_name]

            # Allow handler to customize connector configuration
            if hasattr(handler, "customize_connector_config"):
                connector_config = handler.customize_connector_config(connector_config)

            # Create connector
            connector = StreamableHTTPConnector(connector_config)
            connectors[service_name] = connector

        if not connectors:
            logger.error("No valid services available after checking")
            return None

        if not handler:
            logger.error("No handler available")
            return None

        logger.info(f"🎯 Using {len(connectors)} service(s): {list(connectors.keys())}")

        # Use the unified process_query method with dict of connectors
        # Let exceptions propagate for better error handling
        return await handler.process_query(connectors, query, **kwargs)

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
