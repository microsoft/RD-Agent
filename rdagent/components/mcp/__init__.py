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
    execute_mcp_query_isolated,
    get_service_status,
    is_service_available,
    list_available_mcp_services,
    query_mcp,
    query_mcp_sync,
)

__all__ = [
    # Unified interface
    "query_mcp",
    "query_mcp_sync",
    "execute_mcp_query_isolated",
    # Service management
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
