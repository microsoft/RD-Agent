"""MCP Agent - A user-friendly interface for MCP services

This module provides an Agent-style interface similar to pydantic_ai,
making it easier to use MCP services with a clean API.

Example:
    # Auto-load all configured services
    agent = MCPAgent()
    result = await agent.run("What is 7 plus 5?")

    # Use specific services
    agent = MCPAgent(toolsets=['context7', 'deepwiki'])
    result = agent.run_sync("How to fix ImportError?")
"""

from typing import List, Optional, Union
from urllib.parse import urlparse

from rdagent.components.mcp.registry import MCPServiceConfig, get_global_registry
from rdagent.components.mcp.unified import mcp_execute, mcp_execute_sync
from rdagent.log import rdagent_logger as logger


class MCPServerStreamableHTTP:
    """
    Represents a single MCP service that can be used as a toolset.

    This class encapsulates MCP service configuration and provides
    a clean interface for creating services from URLs or config names.
    """

    def __init__(
        self, url: Optional[str] = None, name: Optional[str] = None, handler: Optional[str] = None, **extra_config
    ):
        """
        Initialize an MCP service.

        Args:
            url: Service URL for dynamic creation
            name: Service name (for identification)
            handler: Handler class specification (e.g., 'module.path:ClassName')
            **extra_config: Additional config including model, api_key, api_base, etc.

        Examples:
            # Create from URL with custom config
            server = MCPServerStreamableHTTP(
                url='http://localhost:8000/mcp',
                model='gpt-4',
                api_key='xxx'
            )

            # Create with minimal config (name will be auto-generated)
            server = MCPServerStreamableHTTP(url='http://localhost:8000/mcp')
        """
        if not url and not name:
            raise ValueError("Either 'url' or 'name' must be provided")

        self.url = url
        self.name = name or self._generate_name_from_url(url) if url else name
        self.handler = handler or "rdagent.components.mcp.general_handler:GeneralMCPHandler"
        self.extra_config = extra_config
        self.enabled = extra_config.pop("enabled", True)
        self.timeout = extra_config.pop("timeout", 120.0)

        # Store as MCPServiceConfig for consistency
        self.config = MCPServiceConfig(
            name=self.name,
            url=self.url or "",
            handler=self.handler,
            enabled=self.enabled,
            timeout=self.timeout,
            extra_config=self.extra_config,
        )

    @classmethod
    def from_config(cls, name: str) -> "MCPServerStreamableHTTP":
        """
        Create an MCP service from mcp_config.json configuration.

        Args:
            name: Service name as defined in mcp_config.json

        Returns:
            MCPServerStreamableHTTP instance

        Raises:
            ValueError: If service not found in configuration

        Example:
            # Load 'context7' service from mcp_config.json
            context7 = MCPServerStreamableHTTP.from_config('context7')
        """
        registry = get_global_registry()
        config = registry.get_service_config(name)

        if not config:
            available = registry.get_enabled_services()
            raise ValueError(f"Service '{name}' not found in configuration. " f"Available services: {available}")

        return cls(
            url=config.url,
            name=config.name,
            handler=config.handler,
            timeout=config.timeout,
            enabled=config.enabled,
            **config.extra_config,
        )

    def _generate_name_from_url(self, url: str) -> str:
        """Generate a service name from URL."""
        # Extract port or path as identifier

        parsed = urlparse(url)
        if parsed.port:
            return f"mcp_{parsed.port}"
        elif parsed.path and parsed.path != "/":
            # Use path as name (remove leading /)
            path_name = parsed.path.lstrip("/").replace("/", "_")
            return f"mcp_{path_name}"
        else:
            return f"mcp_{parsed.hostname or 'service'}"

    def to_service_config(self) -> MCPServiceConfig:
        """Convert to MCPServiceConfig for registry."""
        return self.config

    def __repr__(self) -> str:
        model = self.extra_config.get("model", "default")
        return f"MCPServerStreamableHTTP(name='{self.name}', url='{self.url}', model='{model}')"




class MCPAgent:
    """
    Agent-style interface for MCP services, inspired by pydantic_ai.

    This class provides a clean, user-friendly API for working with MCP services,
    handling service initialization, query execution, and resource management.
    """

    def __init__(
        self,
        toolsets: Optional[Union[List[str], List[MCPServerStreamableHTTP], str, MCPServerStreamableHTTP]] = None,
        **kwargs,
    ):
        """
        Initialize MCPAgent with specified toolsets.

        Args:
            toolsets: MCP services to use:
                - None: Auto-load all enabled services from mcp_config.json
                - str: Single service name from config
                - List[str]: Multiple service names from config
                - MCPServerStreamableHTTP: Single service object
                - List[MCPServerStreamableHTTP]: Multiple service objects
            **kwargs: Additional parameters passed to query_mcp

        Examples:
            # Auto-load all configured services
            agent = MCPAgent()

            # Use specific services from config
            agent = MCPAgent(toolsets=['context7', 'deepwiki'])

            # Use a single service
            agent = MCPAgent(toolsets='context7')

            # Use custom service objects
            custom = MCPServerStreamableHTTP(url='http://localhost:8000/mcp', model='gpt-4')
            agent = MCPAgent(toolsets=[custom])
        """
        self.kwargs = kwargs
        self._services = []
        self._service_names = []

        # Parse toolsets parameter
        if toolsets is None:
            # Auto mode - will use all available services
            self._service_names = None
            logger.info("MCPAgent initialized in auto mode (all available services)", tag="mcp_agent")
        elif isinstance(toolsets, str):
            # Single service name
            self._service_names = [toolsets]
        elif isinstance(toolsets, MCPServerStreamableHTTP):
            # Single service object
            self._register_service(toolsets)
        elif isinstance(toolsets, list):
            for item in toolsets:
                if isinstance(item, str):
                    # Service name from config
                    self._service_names.append(item)
                elif isinstance(item, MCPServerStreamableHTTP):
                    # Service object
                    self._register_service(item)
                else:
                    raise ValueError(f"Invalid toolset item type: {type(item)}")
        else:
            raise ValueError(f"Invalid toolsets type: {type(toolsets)}")

        # Log initialization
        if self._service_names is not None:
            logger.info(f"MCPAgent initialized with services: {self._service_names}", tag="mcp_agent")
        if self._services:
            service_info = [s.name for s in self._services]
            logger.info(f"MCPAgent registered dynamic services: {service_info}", tag="mcp_agent")

    def _register_service(self, service: MCPServerStreamableHTTP):
        """Register a dynamic service with the global registry."""
        self._services.append(service)
        self._service_names = self._service_names or []
        if service.name:
            self._service_names.append(service.name)

            # Register with global registry for this session
            registry = get_global_registry()
            registry.config.mcp_services[service.name] = service.to_service_config()
            logger.info(f"Registered dynamic service: {service.name}", tag="mcp_agent")


    async def run(self, query: str, **kwargs) -> Optional[str]:
        """
        Execute an async query using the configured MCP services.

        Args:
            query: The query to process
            **kwargs: Additional parameters (override instance kwargs)

        Returns:
            Query result or None if failed

        Example:
            agent = MCPAgent(toolsets=['context7'])
            result = await agent.run("How to fix ImportError?")
        """
        # Merge instance kwargs with call kwargs (call kwargs override)
        merged_kwargs = {**self.kwargs, **kwargs}

        # Use mcp_execute with our configured services
        return await mcp_execute(query=query, services=self._service_names, **merged_kwargs)

    def run_sync(self, query: str, **kwargs) -> Optional[str]:
        """
        Execute a synchronous query using the configured MCP services.

        This is convenient for scripts and notebooks that don't use asyncio.

        Args:
            query: The query to process
            **kwargs: Additional parameters (override instance kwargs)

        Returns:
            Query result or None if failed

        Example:
            agent = MCPAgent(toolsets=['context7'])
            result = agent.run_sync("How to fix ImportError?")
        """
        # Merge instance kwargs with call kwargs (call kwargs override)
        merged_kwargs = {**self.kwargs, **kwargs}

        # Use mcp_execute_sync with our configured services
        return mcp_execute_sync(query=query, services=self._service_names, **merged_kwargs)

    # Async context manager support
    async def __aenter__(self):
        """Enter async context (for future extension)."""
        # Currently no special setup needed
        # Could add connection pre-warming in the future
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context (for future extension)."""
        # Currently no cleanup needed
        # Could add connection cleanup in the future
        pass

    def __repr__(self) -> str:
        if self._service_names is None:
            return "MCPAgent(toolsets=<all>)"
        return f"MCPAgent(toolsets={self._service_names})"


# Convenience function for quick one-off queries
def create_agent(toolsets: Optional[Union[List[str], str]] = None, **kwargs) -> MCPAgent:
    """
    Convenience function to create an MCPAgent.

    Args:
        toolsets: Service names to use (None for all)
        **kwargs: Additional configuration

    Returns:
        Configured MCPAgent instance

    Example:
        agent = create_agent(['context7'])
        result = agent.run_sync("What is 2+2?")
    """
    return MCPAgent(toolsets=toolsets, **kwargs)


