import asyncio
import concurrent.futures
import json
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from litellm import RateLimitError
from pydantic import BaseModel, Field, ValidationError

from rdagent.components.mcp.conf import is_mcp_enabled
from rdagent.components.mcp.connector import (
    MCPConnectionError,
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
    handler: Optional[str] = Field(
        default=None,
        description="MCP protocol handler class that processes service communication. "
        "Format: 'module.path:ClassName' (e.g., 'rdagent.components.mcp.context7.handler:Context7Handler'). "
        "If not specified, auto-discovery will attempt to find a handler based on service name.",
    )
    enabled: bool = Field(default=True, description="Whether service is enabled")

    # Service-recommended LLM configuration (can be overridden at Agent or runtime level)
    extra_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Service-recommended LLM configuration (e.g., model, temperature). "
        "These are suggestions based on service capabilities, not requirements. "
        "Can be overridden by Agent initialization or runtime parameters.",
    )

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
                # Don't set a default handler - let auto-discovery handle it
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

    def _import_handler_class(self, handler_spec: str):
        """Dynamically import Handler class using full module path format only.

        Args:
            handler_spec: Handler specification in format 'module.path:ClassName'

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

            # Log import success
            logger.info(f"Successfully imported handler: {handler_spec}")

            return handler_class

        except (ModuleNotFoundError, ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import handler '{handler_spec}': {e}") from e

    async def ensure_initialized(self):
        """Ensure all services are initialized (only executed once)."""
        if not self._initialized:
            await self.auto_register_all_services()
            self._initialized = True
            # Always show which services were actually registered (important system status)
            registered_services = [name for name in self._handlers.keys()]
            if registered_services:
                logger.info(f"MCP registry initialized with services: {registered_services}")
            else:
                logger.info("MCP registry initialized (no services registered)")

    async def auto_register_all_services(self):
        """Auto-register all enabled services from configuration."""
        for name, config in self.config.mcp_services.items():
            if config.enabled and not self.has_handler(name):
                try:
                    # Determine handler spec (explicit or auto-discovered)
                    handler_spec = self._resolve_handler_spec(name, config)

                    # Dynamically import Handler class
                    handler_class = self._import_handler_class(handler_spec)

                    # Create Handler instance with service configuration
                    handler = handler_class(name, service_url=config.url, extra_config=config.extra_config)

                    # Register Handler
                    self._handlers[name] = handler
                    logger.info(f"Auto-registered handler for service '{name}' with class '{handler_spec}'")

                except Exception as e:
                    logger.error(f"Failed to auto-register service '{name}': {e}")

    def _resolve_handler_spec(self, service_name: str, config: MCPServiceConfig) -> str:
        """Resolve handler specification for a service.

        Args:
            service_name: Name of the service
            config: Service configuration

        Returns:
            Full handler specification in format 'module.path:ClassName'
        """
        # If handler is explicitly specified, use it
        if config.handler:
            return config.handler

        # Try to auto-discover client based on service name
        # Convention: ServiceNameClient in service_name.client module

        # 1. Try service-specific client (e.g., context7 -> Context7Client)
        client_class_name = f"{service_name.title().replace('_', '')}Client"

        # Common client locations to check
        handler_specs = [
            f"rdagent.components.mcp.{service_name}.client:{client_class_name}",
            f"rdagent.components.mcp.{service_name}.handler:{client_class_name}",  # Legacy support
            f"rdagent.components.mcp.handlers.{service_name}:{client_class_name}",
            f"rdagent.components.mcp.client:MCPClient",  # Default fallback
        ]

        for spec in handler_specs:
            try:
                module_path, class_name = spec.split(":", 1)
                module = get_module_by_module_path(module_path)
                if hasattr(module, class_name):
                    logger.info(f"Auto-discovered handler for '{service_name}': {spec}")
                    return spec
            except (ImportError, ModuleNotFoundError, AttributeError):
                continue

        # If no specific handler found, use default MCPClient
        logger.info(f"No specific handler found for '{service_name}', using default MCPClient")
        return "rdagent.components.mcp.client:MCPClient"

    async def query(self, query: str, services: Optional[Union[str, List[str]]] = None, **kwargs) -> Optional[str]:
        """
        Execute query using specified MCP services.

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

        # Build connectors for all target services
        connectors = {}
        for service_name in target_services:
            if not self.has_service(service_name):
                logger.warning(f"Service '{service_name}' not configured, skipping")
                continue

            if not self.has_handler(service_name):
                logger.warning(f"No handler for service '{service_name}', skipping")
                continue

            config = self.get_service_config(service_name)
            connector_config = config.to_connector_config()

            # Allow handler to customize connector
            handler = self._handlers[service_name]
            if hasattr(handler, "customize_connector_config"):
                connector_config = handler.customize_connector_config(connector_config)

            connector = StreamableHTTPConnector(connector_config)
            connectors[service_name] = connector

        if not connectors:
            logger.error("No valid services available")
            return None

        # Log which services are being used
        service_names = list(connectors.keys())
        if len(service_names) == 1:
            logger.info(f"🎯 Using service: {service_names[0]}")
        else:
            logger.info(f"🎯 Using {len(service_names)} services: {service_names}")

        # Use the first available handler to process the query
        # This is a current limitation - ideally we'd have a coordinator
        # that can intelligently route between multiple handlers
        first_service = service_names[0]
        handler = self._handlers[first_service]

        # Pass all connectors to the handler
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


# Public API functions (previously in unified.py)


async def mcp_execute(query: str, services: Optional[Union[str, List[str]]] = None, **kwargs) -> Optional[str]:
    """
    Execute MCP query with specified services.

    This is the core execution function that provides a clean, unified way
    to execute queries against MCP services.

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
    if not is_mcp_enabled():
        logger.error("MCP system is globally disabled")
        return None

    try:
        registry = get_global_registry()
        await registry.ensure_initialized()
        return await registry.query(query, services=services, **kwargs)
    except (RateLimitError, MCPConnectionError) as e:
        logger.warning(f"MCP query encountered retryable error: {e}")
        raise
    except Exception as e:
        logger.error(f"MCP query failed with unexpected error: {e}")
        return None


def mcp_execute_sync(
    query: str, services: Optional[Union[str, List[str]]] = None, timeout: float = 180, **kwargs
) -> Optional[str]:
    """
    Synchronous version of mcp_execute for non-async contexts.

    This function runs MCP queries in a separate thread with proper cleanup
    to avoid event loop conflicts.

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
