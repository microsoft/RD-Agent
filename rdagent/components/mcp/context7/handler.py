"""Context7 Handler Implementation

This handler preserves all the experimental results and optimizations
for Context7 service, including prompt templates, timm library special
handling, caching mechanisms, and error handling logic.
"""

import time
from typing import Optional

from rdagent.components.mcp.cache import get_mcp_cache
from rdagent.components.mcp.conf import get_mcp_global_settings
from rdagent.components.mcp.connector import MCPConnectionError, StreamableHTTPConnector
from rdagent.components.mcp.handlers import BaseMCPHandler
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T


def _is_documentation_not_found(result_text: str) -> bool:
    """Check if result indicates documentation not found."""
    error_indicators = ["Failed to fetch documentation", "Error code: 404", "Documentation not found", "404 Not Found"]
    return any(indicator in result_text for indicator in error_indicators)


def _is_connection_error(result_text: str) -> bool:
    """Check if result indicates connection issues."""
    connection_indicators = ["Connection failed", "Timeout", "Network error", "Service unavailable"]
    return any(indicator in result_text for indicator in connection_indicators)


class Context7Handler(BaseMCPHandler):
    """Context7 MCP service handler with all experimental optimizations.

    This handler preserves all the research results including:
    - Enhanced prompt templates for better documentation search
    - Special handling for timm library (3+ mentions trigger)
    - Intelligent error handling for documentation not found cases
    - Caching mechanism for improved performance
    - Multi-round tool calling with OpenAI integration
    - Flexible configuration: extra_config overrides default settings
    """

    def __init__(self, service_name: str = "context7", service_url: str = "http://localhost:8123/mcp", **config):
        super().__init__(service_name, **config)

        # Store MCP service URL (from registry config)
        self.mcp_url = service_url

        # Extract configuration with priority: extra_config > environment variables
        self.api_key = self._resolve_api_key(config)
        self.api_base = self._resolve_api_base(config)
        self.model = self._resolve_model(config)

        # Setup OpenAI client with resolved configuration
        if self.api_key:
            self.setup_openai_client(self.api_key, self.api_base)
        else:
            logger.warning("No API key found for Context7Handler. Service may not work properly.")

    def _resolve_api_key(self, config: dict) -> str:
        """解析API密钥，优先级：extra_config > 环境变量"""
        # 1. 优先使用extra_config中的配置
        extra_config = config.get("extra_config", {})
        if extra_config.get("api_key"):
            return extra_config["api_key"]
        if extra_config.get("openai_api_key"):
            return extra_config["openai_api_key"]

        # 2. 直接从环境变量获取
        return self._get_litellm_api_key()

    def _resolve_api_base(self, config: dict) -> str:
        """解析API基础URL，优先级：extra_config > 环境变量"""
        # 1. 优先使用extra_config中的配置
        extra_config = config.get("extra_config", {})
        if extra_config.get("api_base"):
            return extra_config["api_base"]
        if extra_config.get("openai_api_base"):
            return extra_config["openai_api_base"]

        # 2. 直接从环境变量获取
        return self._get_litellm_api_base()

    def _resolve_model(self, config: dict) -> str:
        """解析模型名称，优先级：extra_config > 默认值"""
        # 1. 优先使用extra_config中的配置
        extra_config = config.get("extra_config", {})
        if extra_config.get("model"):
            return extra_config["model"]

        # 2. 使用默认模型
        return "gpt-4-turbo"

    def _get_litellm_api_key(self) -> str:
        """从LiteLLM/.env环境变量获取API密钥"""
        import os

        # 尝试常见的环境变量名称
        env_vars = ["OPENAI_API_KEY", "AZURE_API_KEY", "ANTHROPIC_API_KEY", "API_KEY"]

        for var in env_vars:
            value = os.getenv(var)
            if value:
                logger.info(f"Using API key from environment variable: {var}")
                return value

        return None

    def _get_litellm_api_base(self) -> str:
        """从LiteLLM/.env环境变量获取API基础URL"""
        import os

        # 尝试常见的环境变量名称
        env_vars = ["OPENAI_API_BASE", "AZURE_API_BASE", "ANTHROPIC_API_BASE", "API_BASE"]

        for var in env_vars:
            value = os.getenv(var)
            if value:
                logger.info(f"Using API base from environment variable: {var}")
                return value

        return None

    async def process_query(
        self,
        connector: StreamableHTTPConnector,
        query: str,
        full_code: Optional[str] = None,
        max_rounds: int = 5,
        verbose: bool = False,
        **kwargs,
    ) -> str:
        """Process Context7 query with all experimental optimizations.

        Args:
            connector: StreamableHTTP connector
            query: The error message or query to search for
            full_code: Complete code context for better understanding
            max_rounds: Maximum number of tool calling rounds
            verbose: Enable verbose logging
            **kwargs: Additional parameters

        Returns:
            Documentation search result as string
        """
        if not self.has_openai_client:
            return "Context7 handler requires OpenAI client configuration"

        # Check cache first
        global_settings = get_mcp_global_settings()
        cache = get_mcp_cache() if global_settings.cache_enabled else None

        if cache:
            cached_result = cache.get_query_result(query)
            if cached_result:
                logger.info("Returning cached Context7 result")
                cache.log_cache_stats()
                return cached_result

        start_time = time.time()

        try:
            async with connector.connect() as session:
                # Log available tools if verbose
                if verbose:
                    tools = await self.get_available_tools(session)
                    logger.info("Available Context7 tools:")
                    for tool in tools:
                        logger.info(f"  - {tool.name}: {tool.description}")

                # Build enhanced query with experimental optimizations
                enhanced_query = self._build_enhanced_query(query, full_code)

                # Use multi-round tool calling with Context7 optimizations
                result = await self._context7_multi_round_calling(session, enhanced_query, max_rounds, verbose)

                # Log timing and result
                self.log_timing("Context7 query", start_time)
                self._log_result(result, verbose)

                # Cache the result
                if cache:
                    cache.set_query_result(query, result)
                    cache.log_cache_stats()

                return result

        except Exception as e:
            logger.error(f"Context7 query processing failed: {e}")
            raise MCPConnectionError(f"Context7 processing failed: {str(e)}") from e

    def _build_enhanced_query(self, error_message: str, full_code: Optional[str] = None) -> str:
        """Build enhanced query using experimental prompt templates."""
        # Build context information using template
        context_info = ""
        if full_code:
            context_info = T("rdagent.components.mcp.context7.prompts.templates:code_context_template").r(
                full_code=full_code
            )

        # Check for timm library special case (experimental optimization)
        timm_trigger = error_message.lower().count("timm") >= 3
        timm_trigger_text = ""
        if timm_trigger:
            timm_trigger_text = T("rdagent.components.mcp.context7.prompts.templates:timm_special_case").r()
            logger.info("Timm library special case triggered")

        # Construct enhanced query using experimental template
        enhanced_query = T("rdagent.components.mcp.context7.prompts.templates:context7_enhanced_query_template").r(
            error_message=error_message, context_info=context_info, timm_trigger_text=timm_trigger_text
        )

        return enhanced_query

    async def _context7_multi_round_calling(
        self, session, enhanced_query: str, max_rounds: int = 5, verbose: bool = False
    ) -> str:
        """Context7 optimized multi-round tool calling."""
        # Get tools and convert to OpenAI format
        tools = await self.get_available_tools(session)
        openai_tools = await self.convert_tools_to_openai_format(tools)

        # Initialize conversation
        messages = [{"role": "user", "content": enhanced_query}]
        model = self.model  # 使用解析后的模型配置

        for round_count in range(1, max_rounds + 1):
            if verbose:
                logger.info(f"Context7 Round {round_count}: Calling OpenAI...")

            response = await self.call_openai_with_tools(messages, openai_tools, model)
            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            # Check if we have tool calls
            if not assistant_message.tool_calls:
                logger.info(f"Context7 final response received in round {round_count}")
                return assistant_message.content

            # Execute tool calls with Context7 error handling
            tool_results = await self._execute_context7_tools(session, assistant_message.tool_calls, verbose)
            messages.extend(tool_results)

            if verbose:
                logger.info(f"Context7 Round {round_count} completed, checking if more tools needed...")

        logger.warning(f"Context7 reached maximum rounds ({max_rounds}), returning last response")
        return messages[-1].get("content", "No response generated")

    async def _execute_context7_tools(self, session, tool_calls, verbose: bool = False):
        """Execute tool calls with Context7 specific error handling."""
        results = []
        tool_names = [tool_call.function.name for tool_call in tool_calls]

        if verbose:
            logger.info(f"Context7 executing {len(tool_calls)} tool(s): {', '.join(tool_names)}")

        for i, tool_call in enumerate(tool_calls, 1):
            if verbose:
                logger.info(f"Context7 Tool {i}: {tool_call.function.name}")

            try:
                import json

                result = await session.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )

                if verbose:
                    logger.info(f"Context7 Tool {i} executed successfully")

                # Extract and handle result with Context7 optimizations
                result_text = result.content[0].text if result.content else ""
                content = self._handle_context7_result(result_text, i)

                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": content,
                    }
                )

            except Exception as e:
                logger.error(f"Context7 Tool {i} failed: {str(e)}")
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error executing Context7 tool: {str(e)}",
                    }
                )

        return results

    def _handle_context7_result(self, result_text: str, tool_index: int) -> str:
        """Handle tool result with Context7 specific error detection."""
        if _is_documentation_not_found(result_text):
            logger.warning(f"Context7 documentation not found for requested library")
            return (
                "Documentation not found for this library. This library may not have detailed "
                "documentation available in the Context7 knowledge base, but you can still provide "
                "general guidance based on the library information from resolve-library-id."
            )

        if _is_connection_error(result_text):
            logger.warning(f"Context7 connection error while fetching documentation")
            return "Connection error occurred while fetching documentation. Please try again later."

        return result_text

    def _log_result(self, result: str, verbose: bool = False):
        """Log result based on verbosity and length."""
        if verbose:
            # In verbose mode, show full result
            logger.info(f"Context7 result: {result}")
        else:
            # In normal mode, show preview if result is long
            if len(result) > 300:
                result_preview = result[:300] + "..."
                logger.info(f"Context7 result preview: {result_preview}")
            else:
                logger.info(f"Context7 result: {result}")

    def get_service_info(self) -> dict:
        """Get Context7 service information."""
        info = super().get_service_info()
        info.update(
            {
                "context7_url": self.mcp_url,
                "model": self.model,  # 使用解析后的模型配置
                "api_key_source": self._get_api_key_source(),  # 显示API密钥来源
                "has_api_key": self.api_key is not None,
                "optimizations": [
                    "Enhanced prompt templates",
                    "Timm library special handling",
                    "Intelligent error handling",
                    "Result caching",
                    "Multi-round tool calling",
                ],
            }
        )
        return info

    def _get_api_key_source(self) -> str:
        """获取API密钥的来源信息，用于调试"""
        if not self.api_key:
            return "None"

        extra_config = self.config.get("extra_config", {})
        if extra_config.get("api_key") or extra_config.get("openai_api_key"):
            return "extra_config"
        else:
            return "environment_variables"
