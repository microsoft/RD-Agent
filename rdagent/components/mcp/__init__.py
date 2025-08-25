"""MCP (Model Context Protocol) integration for RD-Agent.

This module provides a unified interface for various MCP implementations.
Currently supports:
- Context7: Documentation search and error resolution
"""

from .context7 import query_context7

__all__ = ["query_context7"]
