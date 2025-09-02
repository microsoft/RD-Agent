"""MCP Handler Implementation

This module contains both the abstract base class and the general implementation
for MCP service handlers. It provides a unified interface for MCP services using
LiteLLM backend for all LLM calls.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class ConversationCheckpoint:
    """Save multi-round conversation progress for resumption after errors."""

    conversation: List[Dict[str, Any]] = field(default_factory=list)
    completed_rounds: int = 0
    initial_query: Optional[str] = None
    last_tool_results: Optional[Dict[str, Any]] = None
    _last_saved_length: int = 0  # Track last saved position

    def save_round_complete(self, round_num: int, messages: List[Dict[str, Any]]):
        """Save complete conversation state after a round completes.

        Args:
            round_num: The round number that just completed
            messages: Complete conversation history up to this point
        """
        # Replace entire conversation with current state
        self.conversation = messages.copy()
        self.completed_rounds = round_num
        self._last_saved_length = len(messages)

    def can_resume(self) -> bool:
        """Check if we can resume from a checkpoint."""
        return self.completed_rounds > 0 and len(self.conversation) > 0

    def get_resume_messages(self) -> List[Dict[str, Any]]:
        """Get messages to resume conversation from checkpoint."""
        if self.can_resume():
            logger.info(
                f"📂 Resuming from checkpoint: {self.completed_rounds} rounds, {len(self.conversation)} messages",
                tag="mcp_checkpoint",
            )
        return self.conversation.copy()


# TODO: to be tested for multi-service mode
class MultiServiceContext:
    """Context manager for multiple MCP service connections."""

    def __init__(self):
        self.contexts = []  # List of (service_name, context_manager)
        self.sessions = {}  # {service_name: session}
        self.tool_to_service = {}  # {tool_name: service_name}
        self.handlers = {}  # {service_name: handler} - for applying service-specific logic

    async def add_service(
        self, service_name: str, connector: StreamableHTTPConnector, handler=None
    ) -> Optional[MCPSession]:
        """Add a service connection to the context.

        Args:
            service_name: Name of the service
            connector: Connector for the service
            handler: Optional handler for service-specific logic (validation, processing, etc.)

        Returns:
            MCPSession if successful, None otherwise
        """
        try:
            # Connect to the service
            ctx = connector.connect()
            session = await ctx.__aenter__()

            # Store the context and session
            self.contexts.append((service_name, ctx))
            self.sessions[service_name] = session

            # Store handler if provided
            if handler:
                self.handlers[service_name] = handler

            # Map tools to service
            for tool in session.tools:
                self.tool_to_service[tool.name] = service_name

            logger.info(f"✅ Connected '{service_name}' with {len(session.tools)} tools", tag="multi_service")
            return session

        except Exception as e:
            logger.error(f"❌ Failed to connect service '{service_name}': {e}", tag="multi_service")
            return None

    async def cleanup(self):
        """Clean up all service connections."""
        for service_name, ctx in self.contexts:
            try:
                await ctx.__aexit__(None, None, None)
                logger.warning(f"Closed connection to '{service_name}'", tag="multi_service")
            except Exception as e:
                logger.error(f"Error closing '{service_name}': {e}", tag="multi_service")

    def get_all_tools(self) -> List:
        """Get all tools from all connected services."""
        all_tools = []
        for session in self.sessions.values():
            all_tools.extend(session.tools)
        return all_tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: str = None,
        tool_index: int = 1,
        verbose: bool = False,
    ) -> Any:
        """Execute a tool by routing to the correct service with handler-specific logic.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            tool_call_id: OpenAI tool call ID for matching responses
            tool_index: Index of this tool in the current batch (1-based)
            verbose: Enable verbose logging

        Returns:
            Tool execution result with proper formatting
        """
        service_name = self.tool_to_service.get(tool_name)

        if not service_name:
            error_msg = f"Tool '{tool_name}' not found in any service"
            logger.error(error_msg, tag="multi_service")
            result = {"role": "tool", "content": error_msg}
            if tool_call_id:
                result["tool_call_id"] = tool_call_id
            return result

        session = self.sessions.get(service_name)
        if not session:
            error_msg = f"Session for service '{service_name}' not found"
            logger.error(error_msg, tag="multi_service")
            result = {"role": "tool", "content": error_msg}
            if tool_call_id:
                result["tool_call_id"] = tool_call_id
            return result

        handler = self.handlers.get(service_name)

        if verbose:
            logger.info(f"🔧 Routing tool '{tool_name}' to service '{service_name}'", tag="multi_service")

        try:
            # Execute the tool
            result = await session.call_tool(tool_name, arguments)
            result_text = result.content[0].text if result.content else ""

            # Apply handler-specific logic if available
            if handler:
                # 1. Check for rate limit
                if hasattr(handler, "detect_rate_limit"):
                    if handler.detect_rate_limit(result_text):
                        logger.warning(f"🚫 Rate limit detected for tool '{tool_name}'", tag="mcp_rate_limit")
                        # Raise RateLimitError to trigger longer wait time
                        raise RateLimitError(f"Service rate limited: {result_text[:100]}")

                # 2. Validate tool response
                if hasattr(handler, "validate_tool_response"):
                    handler.validate_tool_response(tool_name, result_text)

                # 3. Process result with tool index
                if hasattr(handler, "handle_tool_result"):
                    result_text = handler.handle_tool_result(result_text, tool_name, tool_index)

            # Return with tool_call_id for proper OpenAI matching
            result = {"role": "tool", "content": result_text}
            if tool_call_id:
                result["tool_call_id"] = tool_call_id
            return result

        except (RateLimitError, MCPConnectionError):
            # These exceptions must propagate to trigger retry mechanism
            logger.warning(f"🔄 Tool '{tool_name}' error will trigger retry", tag="multi_service")
            raise
        except Exception as e:
            # Other exceptions are logged but don't trigger retry
            logger.error(f"Error executing tool '{tool_name}': {e}", tag="multi_service")
            result = {"role": "tool", "content": f"Error executing tool: {str(e)}"}
            if tool_call_id:
                result["tool_call_id"] = tool_call_id
            return result


class GeneralMCPHandler(BaseMCPHandler):
    """General MCP service handler using LiteLLM backend.

    This handler provides common functionality for all MCP services:
    - Uses LiteLLM backend for unified model calling and configuration
    - Supports multi-round tool calling with verbose logging
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
            **kwargs: Additional context (e.g., full_code, verbose)

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

    async def process_query_with_resume(
        self,
        connector: StreamableHTTPConnector,
        query: str,
        checkpoint: Optional[ConversationCheckpoint] = None,
        max_retries: int = 3,
        max_rounds: int = 5,
        verbose: bool = False,
        **kwargs,
    ) -> str:
        """Process query with checkpoint-based resumption on errors.

        Args:
            connector: MCP connector
            query: Query string
            checkpoint: Optional checkpoint to resume from
            max_retries: Maximum retry attempts with checkpoint
            max_rounds: Maximum conversation rounds
            verbose: Enable verbose logging
            **kwargs: Additional arguments

        Returns:
            Final response string
        """
        # Initialize checkpoint if not provided
        if checkpoint is None:
            checkpoint = ConversationCheckpoint(initial_query=query)

        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                # Execute the actual processing logic with checkpoint support
                result = await self._process_query_internal(
                    connector=connector,
                    query=query if not checkpoint.can_resume() else checkpoint.initial_query,
                    checkpoint=checkpoint,
                    max_rounds=max_rounds - checkpoint.completed_rounds,  # Adjust remaining rounds
                    verbose=verbose,
                    **kwargs,
                )
                return result

            except (RateLimitError, MCPConnectionError) as e:
                last_error = e
                retry_count += 1

                # Skip retry for real connection errors
                if isinstance(e, MCPConnectionError) and "connection" in str(e).lower():
                    # Connection errors are already retried at connector level, don't retry here
                    raise

                if retry_count <= max_retries:
                    # Determine wait time based on error type
                    if isinstance(e, RateLimitError):
                        # Rate limit: use longer wait times
                        if retry_count <= len(self.RATE_LIMIT_WAIT_TIMES):
                            wait_time = self.RATE_LIMIT_WAIT_TIMES[retry_count - 1]
                        else:
                            wait_time = self.RATE_LIMIT_WAIT_TIMES[-1]  # Use max wait time

                        logger.warning(
                            f"⏳ Rate limited after {checkpoint.completed_rounds} rounds. "
                            f"Waiting {wait_time}s before retry {retry_count}/{max_retries}",
                            tag="mcp_rate_limit",
                        )
                    else:
                        # Normal error: use shorter wait times
                        if retry_count <= len(self.NORMAL_RETRY_WAIT_TIMES):
                            wait_time = self.NORMAL_RETRY_WAIT_TIMES[retry_count - 1]
                        else:
                            wait_time = self.NORMAL_RETRY_WAIT_TIMES[-1]  # Use max wait time

                        logger.warning(
                            f"⚠️ Error after {checkpoint.completed_rounds} rounds: {str(e)[:100]}. "
                            f"Waiting {wait_time}s before retry {retry_count}/{max_retries}",
                            tag="mcp_retry",
                        )

                    # Wait before retry
                    await asyncio.sleep(wait_time)

                    # Continue with checkpoint - will resume from where it left off
                    if checkpoint.can_resume():
                        logger.info(
                            f"📂 Resuming from checkpoint: {checkpoint.completed_rounds} rounds completed",
                            tag="mcp_resume",
                        )
                else:
                    # Max retries reached
                    error_type = "Rate limit" if isinstance(e, RateLimitError) else "Error"
                    logger.error(
                        f"{error_type}: Max retries ({max_retries}) reached. Last error: {last_error}", tag="mcp_retry"
                    )
                    raise

            except Exception as e:
                # Non-retryable error
                logger.error(f"Non-retryable error: {e}", tag="mcp_error")
                raise

        # Should not reach here, but handle it
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected error in retry loop")

    async def _process_query_internal(
        self,
        connector: StreamableHTTPConnector,
        query: str,
        checkpoint: ConversationCheckpoint,
        max_rounds: int = 5,
        verbose: bool = False,
        **kwargs,
    ) -> str:
        """Internal query processing with checkpoint support.

        Args:
            connector: MCP connector
            query: Query string
            checkpoint: Checkpoint for saving progress
            max_rounds: Maximum conversation rounds
            verbose: Enable verbose logging
            **kwargs: Additional arguments

        Returns:
            Final response string
        """
        # Check if backend supports function calling
        if not self.backend.supports_function_calling():
            logger.error("Model does not support function calling, falling back to basic processing", tag="general_mcp")
            return await self._fallback_processing(connector, query, **kwargs)

        # Check cache first (only on fresh start, not resume)
        if not checkpoint.can_resume():
            cached_result = self._check_cache(query)
            if cached_result:
                logger.info("Returning cached result", tag="general_mcp")
                return cached_result

        start_time = time.time()

        try:
            async with connector.connect() as session:
                # Log available tools
                tools = session.tools
                logger.info(f"🔧 Available tools: {[tool.name for tool in tools]}", tag="mcp_session")

                # Preprocess query using subclass implementation
                enhanced_query = self.preprocess_query(query, verbose=verbose, **kwargs)

                # Convert MCP tools to OpenAI format
                openai_tools = self.backend.convert_mcp_tools_to_openai_format(session.tools)

                # Create tool executor for this session
                async def tool_executor(tool_calls):
                    results = await self._execute_session_tools(session, tool_calls, verbose)
                    # Save tool results to checkpoint
                    checkpoint.last_tool_results = {"tool_calls": tool_calls, "results": results}
                    return results

                # Create round callback to save checkpoint after each round
                async def round_callback(round_num: int, messages: List[Dict[str, Any]]):
                    """Save checkpoint after each round completes."""
                    checkpoint.save_round_complete(round_num, messages)

                # Prepare initial messages
                if checkpoint.can_resume():
                    # Resume from checkpoint
                    initial_messages = checkpoint.get_resume_messages()
                    logger.info(f"Resuming with {len(initial_messages)} messages from checkpoint", tag="mcp_resume")
                else:
                    # Fresh start
                    initial_messages = [{"role": "user", "content": enhanced_query}]

                # Perform multi-round tool calling using unified LiteLLM backend
                final_response, conversation = await self.backend.multi_round_tool_calling(
                    initial_messages=initial_messages,
                    tools=openai_tools,
                    max_rounds=max_rounds,
                    tool_executor=tool_executor,
                    verbose=verbose,
                    model_config_override=self.mcp_llm_settings,  # Pass MCP-specific configuration
                    round_callback=round_callback,  # Save progress after each round
                )

                # No need to save here - already saved via round_callback

                # Log timing and result
                self._log_timing("query processing", start_time)

                # Cache the result (only cache final successful results)
                self._cache_result(query, final_response)

                return final_response

        except RateLimitError:
            logger.error(f"Rate limit exceeded for {self.service_name}, will retry automatically", tag="general_mcp")
            raise
        except MCPConnectionError as e:
            # Add context to connection errors
            logger.error(f"MCP connection failed for {self.service_name}: {e}", tag="general_mcp")
            raise
        except Exception as e:
            logger.error(f"MCP query processing failed for {self.service_name}: {str(e)[:200]}", tag="general_mcp")
            # Provide a clean error message to the application layer
            if "timeout" in str(e).lower():
                raise MCPConnectionError("Service timeout - please try again later") from e
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                raise MCPConnectionError("Network connectivity issue - check your connection") from e
            else:
                raise MCPConnectionError("Service temporarily unavailable - please retry") from e

    async def process_query(
        self, connector: StreamableHTTPConnector, query: str, max_rounds: int = 5, verbose: bool = False, **kwargs
    ) -> str:
        """
        Process query with automatic checkpoint-based retry on errors.

        This method automatically handles retries with checkpoint resumption,
        ensuring that successful rounds are not repeated during retries.

        Args:
            connector: StreamableHTTP connector
            query: The query to process
            max_rounds: Maximum number of tool calling rounds
            verbose: Enable verbose logging
            **kwargs: Additional parameters passed to preprocess_query

        Returns:
            Final response string
        """
        # Directly use checkpoint-based processing as the default
        return await self.process_query_with_resume(
            connector=connector,
            query=query,
            checkpoint=None,  # Will be created internally
            max_retries=3,
            max_rounds=max_rounds,
            verbose=verbose,
            **kwargs,
        )

    async def process_multi_services(
        self,
        service_configs: Dict[str, Tuple[StreamableHTTPConnector, Any]],
        query: str,
        max_rounds: int = 5,
        verbose: bool = False,
        **kwargs,
    ) -> Optional[str]:
        """Process multiple MCP services in parallel.

        This method connects to multiple services simultaneously, merges their tools,
        and allows the LLM to choose which tools to use from any service.

        Args:
            service_configs: Dict mapping service names to (connector, handler) tuples
            query: The query to process
            max_rounds: Maximum number of tool calling rounds
            verbose: Enable verbose logging
            **kwargs: Additional parameters

        Returns:
            Final response string or None if failed
        """
        # Create multi-service context
        multi_ctx = MultiServiceContext()

        try:
            # 1. Connect to all services in parallel
            logger.info(f"🔌 Connecting to {len(service_configs)} services...", tag="multi_service")

            connect_tasks = []
            for service_name, (connector, handler) in service_configs.items():
                # Pass handler to add_service for applying service-specific logic
                connect_tasks.append(multi_ctx.add_service(service_name, connector, handler))

            # Wait for all connections
            await asyncio.gather(*connect_tasks)

            # Check if we have any successful connections
            if not multi_ctx.sessions:
                logger.error("No services could be connected", tag="multi_service")
                return None

            # 2. Get all tools
            all_tools = multi_ctx.get_all_tools()

            if not all_tools:
                logger.error("No tools available from any service", tag="multi_service")
                return None

            logger.info(
                f"📊 Connected {len(multi_ctx.sessions)}/{len(service_configs)} services "
                f"with {len(all_tools)} total tools",
                tag="multi_service",
            )

            # Log tool distribution if verbose
            if verbose:
                for service_name, session in multi_ctx.sessions.items():
                    tool_names = [t.name for t in session.tools]
                    logger.info(f"  • {service_name}: {tool_names}", tag="multi_service")

            # 3. Check cache (only for fresh queries)
            cached_result = self._check_cache(query)
            if cached_result:
                logger.info("Returning cached result", tag="multi_service")
                return cached_result

            # 4. Preprocess query
            enhanced_query = self.preprocess_query(query, verbose=verbose, **kwargs)

            # 5. Convert tools to OpenAI format
            openai_tools = self.backend.convert_mcp_tools_to_openai_format(all_tools)

            # 6. Create tool executor that routes to correct service
            async def tool_executor(tool_calls):
                results = []
                for i, tool_call in enumerate(tool_calls, 1):
                    # Use object attributes, not dictionary keys
                    tool_name = tool_call.function.name

                    # Parse arguments
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    try:
                        # Execute tool through multi-service context
                        # Pass tool_call_id and tool_index for proper handling
                        result = await multi_ctx.execute_tool(
                            tool_name, arguments, tool_call_id=tool_call.id, tool_index=i, verbose=verbose
                        )
                        results.append(result)
                    except (RateLimitError, MCPConnectionError):
                        # Critical: propagate these exceptions to trigger retry mechanism
                        logger.warning(
                            f"⚠️ Tool '{tool_name}' triggered retry exception in multi-service mode", tag="multi_service"
                        )
                        raise
                    except Exception as e:
                        # Other exceptions: log but continue with other tools
                        logger.error(f"Tool '{tool_name}' failed: {e}", tag="multi_service")
                        results.append(
                            {"role": "tool", "tool_call_id": tool_call.id, "content": f"Error executing tool: {str(e)}"}
                        )

                return results

            # 7. Perform multi-round tool calling
            start_time = time.time()

            final_response, _ = await self.backend.multi_round_tool_calling(
                initial_messages=[{"role": "user", "content": enhanced_query}],
                tools=openai_tools,
                max_rounds=max_rounds,
                tool_executor=tool_executor,
                verbose=verbose,
                model_config_override=self.mcp_llm_settings,
            )

            # 8. Log timing and cache result
            self._log_timing("multi-service query processing", start_time)
            self._cache_result(query, final_response)

            return final_response

        except Exception as e:
            logger.error(f"Multi-service query failed: {e}", tag="multi_service")
            return None

        finally:
            # Always clean up connections
            await multi_ctx.cleanup()

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

        # Tool execution details reduced

        for i, tool_call in enumerate(tool_calls, 1):
            try:

                # Execute tool call
                result = await session.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )

                # Extract and process result
                result_text = result.content[0].text if result.content else ""

                # Check for rate limit BEFORE validation
                if self.detect_rate_limit(result_text):
                    logger.warning(f"🚫 Rate limit detected for tool '{tool_call.function.name}'", tag="mcp_rate_limit")
                    # Raise RateLimitError to trigger longer wait time
                    raise RateLimitError(f"Service rate limited: {result_text[:100]}")

                # Validate tool response (subclasses can override)
                self.validate_tool_response(tool_call.function.name, result_text)

                processed_content = self.handle_tool_result(result_text, tool_call.function.name, i)

                # Debug logging for tool responses in verbose mode
                if verbose and result_text:
                    preview = result_text[:150].replace("\n", " ")
                    logger.info(f"🔍 Tool '{tool_call.function.name}' response preview: {preview}...", tag="mcp_debug")

                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": processed_content,
                    }
                )

            except RateLimitError:
                # RateLimitError should be propagated to trigger longer retry wait
                logger.warning(
                    f"⏳ Tool {tool_call.function.name} rate limited, propagating for retry with longer wait",
                    tag="mcp_rate_limit",
                )
                raise

            except MCPConnectionError:
                # MCPConnectionError should be propagated to trigger retry mechanism
                logger.warning(
                    f"🔄 Tool {tool_call.function.name} validation failed, propagating for retry", tag="mcp_retry"
                )
                raise

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
