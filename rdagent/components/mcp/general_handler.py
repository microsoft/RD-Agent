"""General MCP Handler Implementation

This handler provides a unified interface for MCP services by using LiteLLM backend
for all LLM calls instead of direct OpenAI client. It extracts common functionality
from specific handlers like Context7Handler to reduce code duplication.
"""

import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from litellm import RateLimitError

from rdagent.components.mcp.cache import get_mcp_cache
from rdagent.components.mcp.conf import get_mcp_global_settings
from rdagent.components.mcp.connector import (
    MCPConnectionError,
    MCPSession,
    StreamableHTTPConnector,
)
from rdagent.components.mcp.handlers import BaseMCPHandler
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.litellm import LITELLM_SETTINGS, LiteLLMAPIBackend


class GeneralMCPHandler(BaseMCPHandler):
    """General MCP service handler using LiteLLM backend.

    This handler provides common functionality for all MCP services:
    - Uses LiteLLM backend for unified model calling and configuration
    - Supports multi-round tool calling with verbose logging
    - Provides customizable hooks for service-specific logic
    - Integrates caching and error handling
    - Maintains compatibility with existing MCP interfaces
    """

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
        for key in ["temperature", "max_tokens", "timeout"]:
            if key in self.extra_config:
                mcp_settings[key] = self.extra_config[key]

        # Log MCP-specific settings
        if mcp_settings:
            # Mask sensitive information
            logged_settings = {k: (v if k != "api_key" else "***") for k, v in mcp_settings.items()}
            logger.info(f"🔧 MCP {self.service_name} LLM config: {logged_settings}", tag="mcp_config")

        return mcp_settings

    # Abstract methods for subclasses to implement

    @abstractmethod
    def preprocess_query(self, query: str, **kwargs) -> str:
        """
        Preprocess the query before sending to LLM.

        Args:
            query: Original query string
            **kwargs: Additional context (e.g., full_code, verbose)

        Returns:
            Enhanced/processed query string
        """
        pass

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
        # Default implementation returns result as-is
        # Subclasses can override for specific error handling
        return result_text

    def should_continue_rounds(self, round_count: int, last_response: str, max_rounds: int) -> bool:
        """
        Determine whether to continue multi-round calling.

        Args:
            round_count: Current round number
            last_response: Last response from LLM
            max_rounds: Maximum allowed rounds

        Returns:
            True to continue, False to stop
        """
        # Default implementation uses round limit only
        return round_count < max_rounds

    # Core implementation using LiteLLM backend

    async def process_query(
        self, connector: StreamableHTTPConnector, query: str, max_rounds: int = 5, verbose: bool = False, **kwargs
    ) -> str:
        """
        Process query using LiteLLM backend with multi-round tool calling.

        Args:
            connector: StreamableHTTP connector
            query: The query to process
            max_rounds: Maximum number of tool calling rounds
            verbose: Enable verbose logging
            **kwargs: Additional parameters passed to preprocess_query

        Returns:
            Final response string
        """
        # Check if backend supports function calling
        if not self.backend.supports_function_calling():
            logger.warning(
                f"Model does not support function calling, falling back to basic processing", tag="general_mcp"
            )
            return await self._fallback_processing(connector, query, **kwargs)

        # Check cache first
        cached_result = self._check_cache(query)
        if cached_result:
            logger.info("Returning cached result", tag="general_mcp")
            return cached_result

        start_time = time.time()

        try:
            async with connector.connect() as session:
                # Log available tools if verbose
                if verbose:
                    tools = session.tools
                    logger.info(f"🔧 Available tools: {[tool.name for tool in tools]}", tag="mcp_session")

                # Preprocess query using subclass implementation
                enhanced_query = self.preprocess_query(query, verbose=verbose, **kwargs)

                # Convert MCP tools to OpenAI format
                openai_tools = self.backend.convert_mcp_tools_to_openai_format(session.tools)

                # Create tool executor for this session
                async def tool_executor(tool_calls):
                    return await self._execute_session_tools(session, tool_calls, verbose)

                # Perform multi-round tool calling using unified LiteLLM backend
                initial_messages = [{"role": "user", "content": enhanced_query}]
                final_response, conversation = await self.backend.multi_round_tool_calling(
                    initial_messages=initial_messages,
                    tools=openai_tools,
                    max_rounds=max_rounds,
                    tool_executor=tool_executor,
                    verbose=verbose,
                    model_config_override=self.mcp_llm_settings,  # Pass MCP-specific configuration
                )

                # Log timing and result
                self._log_timing("query processing", start_time)

                # Cache the result
                self._cache_result(query, final_response)

                return final_response

        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded for {self.service_name}, will retry automatically", tag="general_mcp")
            # Don't re-raise, let the retry mechanism in connector handle it
            raise
        except MCPConnectionError as e:
            # Add context to connection errors
            logger.error(f"MCP connection failed for {self.service_name}: {e}", tag="general_mcp")
            raise
        except Exception as e:
            logger.error(f"MCP query processing failed for {self.service_name}: {str(e)[:200]}", tag="general_mcp")
            # Provide a clean error message to the application layer
            if "timeout" in str(e).lower():
                raise MCPConnectionError(f"Service timeout - please try again later") from e
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                raise MCPConnectionError(f"Network connectivity issue - check your connection") from e
            else:
                raise MCPConnectionError(f"Service temporarily unavailable - please retry") from e

    async def _fallback_processing(self, connector: StreamableHTTPConnector, query: str, **kwargs) -> str:
        """Fallback processing when function calling is not supported."""
        try:
            async with connector.connect() as session:
                if not session.tools:
                    return "No tools available from MCP service"

                # Use first available tool as fallback
                first_tool = session.tools[0]
                result = await session.call_tool(first_tool.name, {"query": query})

                if result.content and len(result.content) > 0:
                    return self.handle_tool_result(result.content[0].text, first_tool.name)
                else:
                    return "No content returned from tool"

        except Exception as e:
            logger.error(f"Fallback processing failed: {e}", tag="general_mcp")
            return f"Error in fallback processing: {str(e)}"

    async def _execute_session_tools(
        self, session: MCPSession, tool_calls: List[Any], verbose: bool = False
    ) -> List[Dict[str, str]]:
        """
        Execute tool calls within an MCP session.

        Args:
            session: Active MCP session
            tool_calls: List of tool calls from LLM
            verbose: Enable verbose logging

        Returns:
            List of tool execution results
        """
        results = []
        tool_names = [tool_call.function.name for tool_call in tool_calls]

        # Tool execution details reduced

        for i, tool_call in enumerate(tool_calls, 1):
            try:
                import json

                # Execute tool call
                result = await session.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )

                # Extract and process result
                result_text = result.content[0].text if result.content else ""
                processed_content = self.handle_tool_result(result_text, tool_call.function.name, i)

                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": processed_content,
                    }
                )

            except Exception as e:
                logger.error(f"❌ Tool {tool_call.function.name} failed: {e}", tag="mcp_error")
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error executing tool: {str(e)}",
                    }
                )

        return results

    # Utility methods

    def _check_cache(self, query: str) -> Optional[str]:
        """Check if query result is cached."""
        global_settings = get_mcp_global_settings()
        if not global_settings.cache_enabled:
            return None

        cache = get_mcp_cache()
        if cache:
            result = cache.get_query_result(query)
            if result:
                cache.log_cache_stats()
            return result
        return None

    def _cache_result(self, query: str, result: str):
        """Cache the query result."""
        global_settings = get_mcp_global_settings()
        if not global_settings.cache_enabled:
            return

        cache = get_mcp_cache()
        if cache:
            cache.set_query_result(query, result)
            cache.log_cache_stats()

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
