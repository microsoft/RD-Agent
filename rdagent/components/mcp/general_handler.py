"""MCP Handler Implementation

This module contains both the abstract base class and the general implementation
for MCP service handlers. It provides a unified interface for MCP services using
LiteLLM backend for all LLM calls.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from litellm import RateLimitError

from rdagent.components.mcp.cache import get_mcp_cache
from rdagent.components.mcp.conf import get_mcp_global_settings
from rdagent.components.mcp.connector import (
    MCPConnectionError,
    MCPSession,
    StreamableHTTPConnector,
)
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.litellm import LiteLLMAPIBackend


class BaseMCPHandler(ABC):
    """Abstract base class for MCP service handlers."""

    def __init__(self, service_name: str, **config):
        """Initialize handler with service name and configuration."""
        self.service_name = service_name
        self.config = config

    @abstractmethod
    async def process_query(self, connector: StreamableHTTPConnector, query: str, **kwargs) -> str:
        """Process a query using the given connector.

        This is the main entry point that each handler must implement.
        It should handle the complete flow from query to response.

        Args:
            connector: The MCP connector to use
            query: The user query to process
            **kwargs: Additional parameters

        Returns:
            The response as a string
        """
        pass

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about this service handler."""
        return {
            "service_name": self.service_name,
            "handler_class": self.__class__.__name__,
            "config_keys": list(self.config.keys()),
        }


class GeneralMCPHandler(BaseMCPHandler):
    """General MCP service handler using LiteLLM backend.

    This handler provides common functionality for all MCP services:
    - Uses LiteLLM backend for unified model calling and configuration
    - Supports multi-round tool calling with detailed logging
    - Provides customizable hooks for service-specific logic
    - Integrates caching and error handling
    - Maintains compatibility with existing MCP interfaces
    """

    # Rate limit wait times (in seconds)
    RATE_LIMIT_WAIT_TIMES = [60, 120, 300]  # 1min, 2min, 5min
    NORMAL_RETRY_WAIT_TIMES = [5, 10, 15]  # 5s, 10s, 15s

    def __init__(self, service_name: str, **config):
        """Initialize handler with LiteLLM backend dependency."""
        super().__init__(service_name, **config)

        # Store extra config from mcp_config.json
        self.extra_config = config.get("extra_config", {})

        # Parse MCP-specific LLM settings
        self.mcp_llm_settings = self._parse_mcp_llm_config()

        # Use standard LiteLLM backend (no global overrides)
        self.backend = LiteLLMAPIBackend()

    def _parse_mcp_llm_config(self) -> dict:
        """Parse MCP service-specific LLM configuration.

        Priority (high to low):
        1. mcp_config.json extra_config (highest priority)
        2. LiteLLM environment variables (fallback)
        3. LiteLLM default settings (lowest priority)

        Returns:
            Dict with service-specific LLM settings
        """
        mcp_settings = {}

        if not self.extra_config:
            return mcp_settings

        # Parse model setting
        if "model" in self.extra_config:
            mcp_settings["model"] = self.extra_config["model"]

        # Parse api_base setting
        if "api_base" in self.extra_config:
            mcp_settings["api_base"] = self.extra_config["api_base"]

        # Parse api_key setting
        if "api_key" in self.extra_config:
            mcp_settings["api_key"] = self.extra_config["api_key"]

        # Parse other LLM settings
        # Note: 'timeout' here only affects LLM API calls (e.g., OpenAI/GPT),
        # NOT MCP tool connections or SSE reading
        for key in ["temperature", "max_tokens", "timeout"]:
            if key in self.extra_config:
                mcp_settings[key] = self.extra_config[key]

        # Log MCP-specific settings
        if mcp_settings:
            # Mask sensitive information
            logged_settings = {k: (v if k != "api_key" else "***") for k, v in mcp_settings.items()}
            logger.info(f"🔧 MCP {self.service_name} LLM config: {logged_settings}", tag="mcp_config")

        return mcp_settings

    # Methods that can be overridden by subclasses for customization

    def preprocess_query(self, query: str, **kwargs) -> str:
        """
        Preprocess the query before sending to LLM.

        Default implementation returns query as-is.
        Subclasses should override this for custom preprocessing.

        Args:
            query: Original query string
            **kwargs: Additional context (e.g., full_code)

        Returns:
            Enhanced/processed query string
        """
        # Default: no preprocessing
        return query

    def handle_tool_result(self, result_text: str, tool_name: str, tool_index: int = 1) -> str:
        """
        Process tool execution result.

        Args:
            result_text: Raw result from tool execution
            tool_name: Name of the executed tool
            tool_index: Index of the tool in current round

        Returns:
            Processed result content
        """
        # Default: return result as-is
        # Subclasses can override for custom processing
        return result_text

    def detect_rate_limit(self, response_text: str) -> bool:
        """
        Detect if response indicates rate limiting.

        Subclasses can override to add service-specific detection patterns.

        Args:
            response_text: The response text to check

        Returns:
            True if rate limit detected, False otherwise
        """
        if not response_text:
            return False

        # Common rate limit patterns
        patterns = [
            "rate limit",
            "too many requests",
            "try again later",
            "try again",
            "429",
            "exceeded the rate",
            "rate-limited",
            "rate limit exceeded",
        ]

        response_lower = response_text.lower()
        return any(pattern in response_lower[:70] for pattern in patterns)

    def validate_tool_response(self, tool_name: str, response_text: str) -> None:
        """
        Validate tool response - subclasses can override.

        Default implementation does no validation.
        Subclasses can override this method to implement custom validation logic.

        Args:
            tool_name: Name of the tool that was called
            response_text: The response text from the tool

        Raises:
            MCPConnectionError: When validation fails, triggering checkpoint-based retry
        """
        # Default: no validation
        # Subclasses override this to add validation logic
        pass

    # Core implementation using LiteLLM backend

    async def _process_query_unified(
        self,
        connectors: Dict[str, StreamableHTTPConnector],
        query: str,
        max_rounds: int = 5,
        **kwargs,
    ) -> str:
        """
        Unified internal query processing for any number of services.

        This method treats single-service as N=1 of multi-service,
        eliminating code duplication.

        Args:
            connectors: Dict mapping service names to connectors
            query: Query string
            max_rounds: Maximum conversation rounds
            **kwargs: Additional arguments

        Returns:
            Final response string
        """
        # Check if backend supports function calling
        if not self.backend.supports_function_calling():
            logger.error("Model does not support function calling", tag="mcp")
            return "Model does not support function calling"

        # Check cache first
        cached_result = self._check_cache(query)
        if cached_result:
            logger.warning("Returning cached result", tag="mcp")
            return cached_result

        start_time = time.time()

        # Use AsyncExitStack to manage all connections
        async with AsyncExitStack() as stack:
            try:
                # Connect to all services
                num_services = len(connectors)
                logger.info(f"🔌 Connecting to {num_services} service(s)...", tag="mcp")

                sessions = {}  # {service_name: session}
                tool_to_service = {}  # {tool_name: service_name}
                failed_services = []

                # Define async function to connect a single service
                async def connect_service(service_name: str, connector):
                    """Connect to a single service and return result."""
                    try:
                        session = await stack.enter_async_context(connector.connect())
                        return service_name, session, None
                    except Exception as e:
                        return service_name, None, e

                # Create connection tasks for all services
                connection_tasks = [
                    connect_service(service_name, connector) for service_name, connector in connectors.items()
                ]

                # Connect to all services concurrently
                connection_results = await asyncio.gather(*connection_tasks)

                # Process connection results
                for service_name, session, error in connection_results:
                    if session is not None:
                        sessions[service_name] = session

                        # Map tools to service
                        for tool in session.tools:
                            tool_to_service[tool.name] = service_name

                        logger.info(f"✅ Connected '{service_name}' with {len(session.tools)} tools", tag="mcp")
                    else:
                        failed_services.append(service_name)
                        logger.warning(f"Service '{service_name}' connection failed: {error}", tag="mcp")

                if not sessions:
                    # All connections failed
                    if failed_services:
                        service_list = ", ".join(failed_services)
                        logger.error(f"Could not connect to any MCP services. Failed: {service_list}", tag="mcp")
                        return (
                            f"Failed to connect to MCP services ({service_list}). Please check if services are running."
                        )
                    else:
                        logger.error("No services could be connected", tag="mcp")
                        return "Failed to connect to services"

                # Get all tools from all sessions
                all_tools = []
                for session in sessions.values():
                    all_tools.extend(session.tools)

                if not all_tools:
                    logger.error("No tools available from services", tag="mcp")
                    return "No tools available"

                logger.info(
                    f"📊 Connected {len(sessions)}/{num_services} service(s) with {len(all_tools)} total tools",
                    tag="mcp",
                )

                # Preprocess query
                enhanced_query = self.preprocess_query(query, **kwargs)

                # Convert tools to OpenAI format
                openai_tools = self.backend.convert_mcp_tools_to_openai_format(all_tools)

                # Create tool executor
                async def tool_executor(tool_calls):
                    """Execute tools, routing to correct service"""
                    results = []

                    for i, tool_call in enumerate(tool_calls, 1):
                        tool_name = tool_call.function.name
                        service_name = tool_to_service.get(tool_name)

                        if not service_name:
                            error_msg = f"Tool '{tool_name}' not found in any service"
                            logger.error(error_msg, tag="mcp")
                            results.append({"role": "tool", "tool_call_id": tool_call.id, "content": error_msg})
                            continue

                        session = sessions.get(service_name)
                        if not session:
                            error_msg = f"Session for service '{service_name}' not found"
                            logger.error(error_msg, tag="mcp")
                            results.append({"role": "tool", "tool_call_id": tool_call.id, "content": error_msg})
                            continue

                        logger.info(f"🔧 Routing tool '{tool_name}' to service '{service_name}'", tag="mcp")

                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            arguments = {}

                        try:
                            # Execute the tool
                            result = await session.call_tool(tool_name, arguments)
                            result_text = result.content[0].text if result.content else ""

                            # Apply handler-specific logic
                            # 1. Check for rate limit
                            if self.detect_rate_limit(result_text):
                                logger.warning(f"🚫 Rate limit detected for tool '{tool_name}'", tag="mcp_rate_limit")
                                raise RateLimitError(f"Service rate limited: {result_text[:100]}")

                            # 2. Validate tool response
                            self.validate_tool_response(tool_name, result_text)

                            # 3. Process result
                            result_text = self.handle_tool_result(result_text, tool_name, i)

                            results.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})
                        except (RateLimitError, MCPConnectionError):
                            # Propagate for retry
                            logger.warning(f"Tool '{tool_name}' triggered retry", tag="mcp")
                            raise
                        except Exception as e:
                            logger.error(f"Tool '{tool_name}' failed: {e}", tag="mcp")
                            results.append(
                                {"role": "tool", "tool_call_id": tool_call.id, "content": f"Error: {str(e)}"}
                            )

                    return results

                # Perform multi-round tool calling
                final_response, _ = await self.backend.multi_round_tool_calling(
                    initial_messages=[{"role": "user", "content": enhanced_query}],
                    tools=openai_tools,
                    max_rounds=max_rounds,
                    tool_executor=tool_executor,
                    model_config_override=self.mcp_llm_settings,
                )

                # Log timing and cache result
                self._log_timing("query processing", start_time)
                self._cache_result(query, final_response)

                return final_response

            except Exception as e:
                logger.error(f"Query processing failed: {str(e)[:200]}", tag="mcp")
                if "timeout" in str(e).lower():
                    raise MCPConnectionError("Service timeout - please try again later") from e
                elif "network" in str(e).lower() or "connection" in str(e).lower():
                    raise MCPConnectionError("Network connectivity issue - check your connection") from e
                else:
                    raise MCPConnectionError("Service temporarily unavailable - please retry") from e

    async def process_query(
        self,
        connectors: Dict[str, StreamableHTTPConnector],
        query: str,
        max_rounds: int = 5,
        **kwargs,
    ) -> str:
        """
        Unified query processing for single or multiple services.

        This method handles both single-service and multi-service scenarios uniformly,
        treating single-service as a special case of multi-service where N=1.

        Args:
            connectors: Dict of service_name -> connector (can have 1 or N entries)
            query: The query to process
            max_rounds: Maximum number of tool calling rounds
            **kwargs: Additional parameters passed to preprocess_query

        Returns:
            Final response string
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                return await self._process_query_unified(
                    connectors=connectors,
                    query=query,
                    max_rounds=max_rounds,
                    **kwargs,
                )
            except (RateLimitError, MCPConnectionError) as e:
                if attempt == max_retries - 1:
                    # Last attempt, re-raise the exception
                    raise

                # Calculate exponential backoff wait time
                wait_time = 2**attempt * 5  # 5s, 10s, 20s

                logger.warning(
                    f"⏳ Query failed (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}. "
                    f"Waiting {wait_time}s before retry",
                    tag="mcp_retry",
                )

                await asyncio.sleep(wait_time)

    # Utility methods

    def _check_cache(self, query: str) -> Optional[str]:
        """Check if query result is cached."""
        global_settings = get_mcp_global_settings()
        if not global_settings.cache_enabled:
            return None

        cache = get_mcp_cache()
        if cache:
            return cache.get_query_result(query)
        return None

    def _cache_result(self, query: str, result: str):
        """Cache the query result."""
        global_settings = get_mcp_global_settings()
        if not global_settings.cache_enabled:
            return

        cache = get_mcp_cache()
        if cache:
            cache.set_query_result(query, result)

    def _log_timing(self, operation: str, start_time: float):
        """Log operation timing."""
        duration = time.time() - start_time
        if duration > 2.0:  # Only log slow operations
            logger.info(f"⏱️ {self.service_name} {operation} took {duration:.1f}s", tag="mcp_timing")

    def get_service_info(self) -> Dict[str, Any]:
        """Get service information - core details only."""
        info = super().get_service_info()
        info.update(
            {
                "backend_type": "LiteLLM",
                "supports_function_calling": self.backend.supports_function_calling(),
            }
        )
        return info
