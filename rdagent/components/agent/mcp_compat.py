"""Compatibility shim for the pydantic-ai MCP HTTP server class.

pydantic-ai renamed ``MCPServerStreamableHTTP`` to ``MCPServerHTTP`` during the
1.x line; both names are still exported there, but newer 1.x versions only
document the new name. The 2.x line reworked the MCP API entirely
(``MCPToolsetClient``) and requires a separate migration, so RD-Agent keeps
``pydantic-ai-slim<2`` in ``requirements.txt``.

Importing through this module keeps the agent components working across every
1.x release regardless of which name is available.
"""

from __future__ import annotations

try:
    from pydantic_ai.mcp import MCPServerHTTP
except ImportError:  # pragma: no cover - only hit on old pydantic-ai 1.x
    from pydantic_ai.mcp import MCPServerStreamableHTTP as MCPServerHTTP

__all__ = ["MCPServerHTTP"]
