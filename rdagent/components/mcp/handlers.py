"""MCP Handler Interface and Base Implementation

This module defines the interface and base implementation for MCP service handlers.
Each MCP service should implement a handler that inherits from BaseMCPHandler.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from mcp.types import Tool
from openai import AsyncOpenAI

from rdagent.components.mcp.connector import MCPSession, StreamableHTTPConnector
from rdagent.log import rdagent_logger as logger


class BaseMCPHandler(ABC):
    """Abstract base class for MCP service handlers."""

    def __init__(self, service_name: str, **config):
        """Initialize handler with service name and configuration."""
        self.service_name = service_name
        self.config = config

        # Optional OpenAI client for tool calling
        self._openai_client: Optional[AsyncOpenAI] = None

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

    def setup_openai_client(self, api_key: str, api_base: Optional[str] = None):
        """Setup OpenAI client for tool calling."""
        self._openai_client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    @property
    def has_openai_client(self) -> bool:
        """Check if OpenAI client is configured."""
        return self._openai_client is not None

    # Helper methods for common MCP operations

    async def get_available_tools(self, session: MCPSession) -> List[Tool]:
        """Get available tools from MCP session."""
        return session.tools

    async def convert_tools_to_openai_format(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools
        ]

    async def call_openai_with_tools(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], model: str = "gpt-4-turbo"
    ) -> Any:
        """Call OpenAI API with tools."""
        if not self.has_openai_client:
            raise RuntimeError("OpenAI client not configured. Call setup_openai_client() first.")

        return await self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

    async def execute_tool_calls(self, session: MCPSession, tool_calls: List[Any]) -> List[Dict[str, str]]:
        """Execute tool calls and return results."""
        results = []

        for tool_call in tool_calls:
            try:
                import json

                result = await session.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )

                # Extract text content from result
                result_text = ""
                if result.content and len(result.content) > 0:
                    result_text = result.content[0].text

                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )

                logger.info(f"Tool {tool_call.function.name} executed successfully")

            except Exception as e:
                logger.error(f"Tool {tool_call.function.name} failed: {e}")
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error executing tool: {str(e)}",
                    }
                )

        return results

    async def multi_round_tool_calling(
        self, session: MCPSession, initial_query: str, model: str = "gpt-4-turbo", max_rounds: int = 5
    ) -> str:
        """Perform multi-round tool calling with OpenAI."""
        if not self.has_openai_client:
            raise RuntimeError("OpenAI client not configured")

        # Get tools and convert to OpenAI format
        tools = await self.get_available_tools(session)
        openai_tools = await self.convert_tools_to_openai_format(tools)

        # Initialize conversation
        messages = [{"role": "user", "content": initial_query}]

        for round_count in range(1, max_rounds + 1):
            logger.info(f"Round {round_count}: Calling OpenAI...")

            response = await self.call_openai_with_tools(messages, openai_tools, model)
            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            # Check if we have tool calls
            if not assistant_message.tool_calls:
                logger.info(f"Final response received in round {round_count}")
                return assistant_message.content

            # Execute tool calls
            tool_results = await self.execute_tool_calls(session, assistant_message.tool_calls)
            messages.extend(tool_results)

            logger.info(f"Round {round_count} completed, continuing...")

        logger.warning(f"Reached maximum rounds ({max_rounds}), returning last response")
        return messages[-1].get("content", "No response generated")

    def log_timing(self, operation: str, start_time: float):
        """Log operation timing."""
        duration = time.time() - start_time
        logger.info(f"{self.service_name} {operation} completed in {duration:.2f}s")


class SimpleQueryHandler(BaseMCPHandler):
    """Simple handler for basic query processing without OpenAI integration."""

    async def process_query(self, connector: StreamableHTTPConnector, query: str, **kwargs) -> str:
        """Process query using direct MCP tool calls."""
        start_time = time.time()

        try:
            async with connector.connect() as session:
                tools = await self.get_available_tools(session)

                if not tools:
                    return "No tools available from MCP service"

                # For demonstration, call the first available tool
                # Real implementations should have smarter tool selection
                if tools:
                    first_tool = tools[0]
                    try:
                        result = await session.call_tool(first_tool.name, {"query": query})

                        if result.content and len(result.content) > 0:
                            response = result.content[0].text
                        else:
                            response = "No content returned from tool"

                        self.log_timing("query", start_time)
                        return response

                    except Exception as e:
                        logger.error(f"Tool call failed: {e}")
                        return f"Error calling tool: {str(e)}"

                return "No suitable tool found for query"

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return f"Error processing query: {str(e)}"


class OpenAIToolCallingHandler(BaseMCPHandler):
    """Handler that uses OpenAI for intelligent tool calling."""

    def __init__(self, service_name: str, **config):
        super().__init__(service_name, **config)

        # Setup OpenAI client from config
        api_key = config.get("api_key") or config.get("openai_api_key")
        api_base = config.get("api_base") or config.get("openai_api_base")

        if api_key:
            self.setup_openai_client(api_key, api_base)

    async def process_query(
        self, connector: StreamableHTTPConnector, query: str, max_rounds: int = 5, model: str = None, **kwargs
    ) -> str:
        """Process query using OpenAI with tool calling."""
        if not self.has_openai_client:
            return "OpenAI client not configured for this handler"

        start_time = time.time()

        try:
            async with connector.connect() as session:
                # Use configured model or default
                model = model or self.config.get("model", "gpt-4-turbo")

                result = await self.multi_round_tool_calling(session, query, model, max_rounds)

                self.log_timing("query", start_time)
                return result

        except Exception as e:
            logger.error(f"OpenAI tool calling failed: {e}")
            return f"Error processing query with OpenAI: {str(e)}"
