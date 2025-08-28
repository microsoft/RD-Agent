"""Streamable HTTP MCP Connector

This module provides a simplified MCP connector specifically for streamable_http protocol.
It abstracts the connection management, session initialization, and tool calling logic.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rdagent.log import rdagent_logger as logger


class StreamableHTTPConfig(BaseModel):
    """Configuration for streamable HTTP MCP connection."""

    url: str = Field(..., description="MCP service URL")
    timeout: float = Field(default=3.0, description="Connection timeout in seconds")
    sse_read_timeout: float = Field(default=300.0, description="SSE read timeout in seconds")
    terminate_on_close: bool = Field(default=True, description="Terminate session on close")
    headers: Optional[Dict[str, Any]] = Field(default=None, description="HTTP headers")

    def model_post_init(self, __context) -> None:
        """Validate URL format."""
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL format: {self.url}. Must start with http:// or https://")


class MCPConnectionError(Exception):
    """Raised when MCP connection fails."""

    pass


class MCPSession:
    """Wrapper for ClientSession with connection info."""

    def __init__(self, session: ClientSession, config: StreamableHTTPConfig):
        self.session = session
        self.config = config
        self._tools: List[Tool] = []

    async def initialize(self) -> None:
        """Initialize session and load available tools."""
        await self.session.initialize()
        tools_result = await self.session.list_tools()
        self._tools = tools_result.tools if tools_result else []

    @property
    def tools(self) -> List[Tool]:
        """Get available tools."""
        return self._tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool with given arguments."""
        return await self.session.call_tool(name, arguments)


class StreamableHTTPConnector:
    """Streamable HTTP MCP connector with simplified interface."""

    def __init__(self, config: StreamableHTTPConfig):
        self.config = config
        self._connection_context = None

    @asynccontextmanager
    async def connect(self):
        """Connect to MCP server and yield session."""
        session = None
        connection_context = None

        try:
            # Create connection context
            connection_context = self._create_connection_context()

            # Establish connection
            streams = await connection_context.__aenter__()
            read_stream, write_stream = streams[:2]  # Handle extra returns safely

            # Create and initialize session
            client_session = ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
                read_timeout_seconds=timedelta(seconds=self.config.timeout),
            )

            await client_session.__aenter__()
            session = MCPSession(client_session, self.config)
            await session.initialize()

            logger.info(f"🔗 Connected: {self.config.url} ({len(session.tools)} tools)", tag="mcp_connection")

            yield session

        except Exception as e:
            # Convert complex exceptions to simpler ones
            error_msg = self._simplify_connection_error(e)
            logger.error(f"MCP connection failed: {error_msg}")
            raise MCPConnectionError(error_msg) from e

        finally:
            # Cleanup resources
            try:
                if session and session.session:
                    await session.session.__aexit__(None, None, None)
            except Exception:
                pass

            try:
                if connection_context:
                    await connection_context.__aexit__(None, None, None)
            except Exception:
                pass

    def _create_connection_context(self):
        """Create streamable HTTP connection context."""
        try:
            # Try with newer MCP versions that support httpx_client_factory
            from mcp.shared._httpx_utils import create_mcp_http_client

            return streamablehttp_client(
                url=self.config.url,
                headers=self.config.headers,
                timeout=timedelta(seconds=self.config.timeout),
                sse_read_timeout=timedelta(seconds=self.config.sse_read_timeout),
                terminate_on_close=self.config.terminate_on_close,
                httpx_client_factory=create_mcp_http_client,
            )
        except (ImportError, TypeError):
            # Fall back to basic call for older versions
            return streamablehttp_client(
                url=self.config.url,
                headers=self.config.headers,
                timeout=timedelta(seconds=self.config.timeout),
                sse_read_timeout=timedelta(seconds=self.config.sse_read_timeout),
                terminate_on_close=self.config.terminate_on_close,
            )

    def _simplify_connection_error(self, error: Exception) -> str:
        """Convert complex MCP connection errors to simple messages."""
        error_str = str(error).lower()

        if "timeout" in error_str:
            return f"Connection timeout after {self.config.timeout}s to {self.config.url}"
        elif "connection" in error_str or "network" in error_str:
            return f"Failed to connect to MCP server at {self.config.url}"
        elif "404" in error_str:
            return f"MCP endpoint not found at {self.config.url}"
        elif "403" in error_str or "401" in error_str:
            return f"Authentication failed for {self.config.url}"
        else:
            return f"MCP connection failed for {self.config.url}: {str(error)[:100]}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=3, max=20),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, RuntimeError, OSError)),
)
async def create_mcp_session_with_retry(config: StreamableHTTPConfig) -> StreamableHTTPConnector:
    """Create MCP session with retry mechanism."""
    return StreamableHTTPConnector(config)
