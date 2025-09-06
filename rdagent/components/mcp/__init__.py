"""MCP (Model Context Protocol) integration for RD-Agent - Clean and Simple

This module provides a unified, clean interface for MCP services with
no legacy compatibility. All dead code has been removed.

New unified interface:
- MCPAgent: Agent-style interface for easy use
- mcp_execute: Core execution function for business code
- MCPServerStreamableHTTP: Service configuration wrapper

Architecture:
- StreamableHTTP connector for MCP communication
- Registry-based service management with JSON configuration
- Handler-based processing with service-specific optimizations
- No backward compatibility - clean slate design
"""

# Main user interfaces
from .agent import MCPAgent, MCPServerStreamableHTTP, create_agent

# Core components (for advanced usage only)
from .connector import StreamableHTTPConfig, StreamableHTTPConnector
from .context7.handler import Context7Handler
from .general_handler import BaseMCPHandler
from .registry import MCPRegistry, MCPRegistryConfig, MCPServiceConfig

# Core execution function for business code
from .unified import mcp_execute, mcp_execute_sync

__all__ = [
    # Main user interfaces (recommended)
    "MCPAgent",
    "MCPServerStreamableHTTP",
    "create_agent",
    # Core execution functions
    "mcp_execute",
    "mcp_execute_sync",
    # Advanced components (for internal use)
    "StreamableHTTPConnector",
    "StreamableHTTPConfig",
    "BaseMCPHandler",
    "Context7Handler",
    "MCPRegistry",
    "MCPRegistryConfig",
    "MCPServiceConfig",
]
