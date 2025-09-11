"""Streamable HTTP MCP Connector

This module provides a simplified MCP connector specifically for streamable_http protocol.
It abstracts the connection management, session initialization, and tool calling logic.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from litellm import RateLimitError
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared._httpx_utils import create_mcp_http_client
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
    timeout: float = Field(default=120.0, description="Connection timeout in seconds")
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Connection retry {retry_state.attempt_number}/3, waiting {retry_state.next_action.sleep}s"
        ),
    )
    @asynccontextmanager
    async def connect(self):
        """Connect to MCP server and yield session with automatic retry on connection errors."""
        session = None
        connection_context = None
        connection_entered = False
        session_entered = False

        try:
            # Create connection context
            connection_context = self._create_connection_context()

            # Establish connection
            try:
                streams = await connection_context.__aenter__()
                connection_entered = True
            except asyncio.CancelledError:
                # Task was cancelled, re-raise immediately
                raise
            except Exception as e:
                # Connection failed at network level
                # Don't try to clean up - let it be garbage collected
                # Trying to clean up causes anyio cross-task errors
                connection_context = None  # Release reference

                error_msg = self._simplify_connection_error(e)
                logger.error(f"MCP connection failed: {error_msg}")
                raise MCPConnectionError(error_msg) from e

            read_stream, write_stream = streams[:2]  # Handle extra returns safely

            # Create and initialize session
            client_session = ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
                read_timeout_seconds=timedelta(seconds=self.config.timeout),
            )

            await client_session.__aenter__()
            session_entered = True
            session = MCPSession(client_session, self.config)

            try:
                await session.initialize()
            except asyncio.CancelledError:
                # Re-raise cancellation without wrapping
                raise
            except Exception as e:
                # Initialize failed - provide clear error
                error_msg = self._simplify_connection_error(e)
                logger.error(f"MCP session initialization failed: {error_msg}")
                raise MCPConnectionError(error_msg) from e

            logger.info(f"🔗 Connected: {self.config.url} ({len(session.tools)} tools)", tag="mcp_connection")

            yield session

        except MCPConnectionError:
            # MCPConnectionError should propagate as-is (could be validation error)
            # Don't re-wrap to preserve original error message for retry logic
            raise
        except asyncio.CancelledError:
            # Task was cancelled, don't wrap in MCPConnectionError
            logger.warning(f"Connection to {self.config.url} was cancelled", tag="mcp_connection")
            raise
        except Exception as e:
            # Unexpected error - convert to MCPConnectionError
            error_msg = self._simplify_connection_error(e)
            logger.error(f"Unexpected MCP error: {error_msg}")
            raise MCPConnectionError(error_msg) from e

        finally:
            # Cleanup resources - only clean up what was successfully entered
            if session_entered and session and session.session:
                try:
                    await session.session.__aexit__(None, None, None)
                except Exception:
                    # Silently ignore cleanup errors
                    pass

            # Only call __aexit__ if we successfully entered the context
            if connection_entered and connection_context:
                try:
                    await connection_context.__aexit__(None, None, None)
                except Exception:
                    # Silently ignore cleanup errors
                    pass

    def _create_connection_context(self):
        """Create streamable HTTP connection context."""
        return streamablehttp_client(
            url=self.config.url,
            headers=self.config.headers,
            timeout=timedelta(seconds=self.config.timeout),
            sse_read_timeout=timedelta(seconds=self.config.sse_read_timeout),
            terminate_on_close=self.config.terminate_on_close,
            httpx_client_factory=create_mcp_http_client,
        )

    def _simplify_connection_error(self, error: Exception) -> str:
        """Convert complex MCP connection errors to simple messages."""
        error_str = str(error).lower()

        # Extract URL components for clearer messages
        parsed = urlparse(self.config.url)
        host_port = f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname

        # Handle rate limit errors specifically
        if isinstance(error, RateLimitError) or "ratelimiterror" in error_str or "rate limit" in error_str:
            return f"Rate limit exceeded for {self.config.url}. Will retry with exponential backoff."
        elif "all connection attempts failed" in error_str or "connection refused" in error_str:
            return f"Cannot connect to {host_port} - service is not running or port is incorrect"
        elif "timeout" in error_str:
            return f"Connection timeout after {self.config.timeout}s to {host_port}"
        elif "connection" in error_str or "network" in error_str:
            return f"Network error connecting to {host_port}"
        elif "404" in error_str:
            return f"MCP endpoint not found at {self.config.url}"
        elif "403" in error_str or "401" in error_str:
            return f"Authentication failed for {self.config.url}"
        elif "cancelled" in error_str:
            return f"Connection attempt was cancelled"
        else:
            # For other errors, try to extract the most relevant part
            error_msg = str(error)
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            return f"MCP error for {host_port}: {error_msg}"
