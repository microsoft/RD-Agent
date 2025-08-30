"""MCP (Model Context Protocol) integration for RD-Agent.

This module provides a unified interface for various MCP implementations with
support for multiple MCP services through a configuration-driven registry system.

Currently supports:
- Context7: Documentation search and error resolution with optimized prompts
- Extensible architecture for adding new MCP services

Architecture:
- StreamableHTTP connector for MCP communication
- Registry-based service management with JSON configuration
- Handler-based processing with service-specific optimizations
- Backward compatibility with existing APIs
"""

from .connector import StreamableHTTPConfig, StreamableHTTPConnector
from .context7.handler import Context7Handler
from .general_handler import BaseMCPHandler

# Core components (for advanced usage)
from .registry import MCPRegistry, MCPRegistryConfig, MCPServiceConfig

# Unified MCP interface
from .unified import (
    get_service_status,
    initialize_mcp_registry,
    is_service_available,
    list_available_mcp_services,
    query_mcp,
    query_mcp_auto,
    register_mcp_handler,
)

__all__ = [
    # Unified interface
    "query_mcp",
    "query_mcp_auto",
    # Service management
    "initialize_mcp_registry",
    "register_mcp_handler",
    "list_available_mcp_services",
    "is_service_available",
    "get_service_status",
    # Core components
    "MCPRegistry",
    "MCPRegistryConfig",
    "MCPServiceConfig",
    "StreamableHTTPConnector",
    "StreamableHTTPConfig",
    "BaseMCPHandler",
    "Context7Handler",
]
