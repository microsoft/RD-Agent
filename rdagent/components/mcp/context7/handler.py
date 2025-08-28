"""Context7 Handler Implementation

This handler preserves all the experimental results and optimizations
for Context7 service, including prompt templates, timm library special
handling, caching mechanisms, and error handling logic.

Now inherits from GeneralMCPHandler to use unified LiteLLM backend.
"""

from typing import Optional

from rdagent.components.mcp.general_handler import GeneralMCPHandler
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


class Context7Handler(GeneralMCPHandler):
    """Context7 MCP service handler with all experimental optimizations.

    This handler preserves all the research results including:
    - Enhanced prompt templates for better documentation search
    - Special handling for timm library (3+ mentions trigger)
    - Intelligent error handling for documentation not found cases
    - All common functionality now handled by GeneralMCPHandler + LiteLLM
    """

    def __init__(self, service_name: str = "context7", service_url: str = "http://localhost:8123/mcp", **config):
        # Initialize with GeneralMCPHandler - uses LiteLLM backend
        super().__init__(service_name, **config)

        # Store MCP service URL (from registry config)
        self.mcp_url = service_url

        # Log that we're using LiteLLM backend
        # Context7Handler initialized with LiteLLM backend

    # Configuration is now handled by LiteLLM backend - no need for separate resolution

    # process_query is now handled by GeneralMCPHandler - we implement the abstract methods

    def preprocess_query(self, query: str, **kwargs) -> str:
        """Preprocess query with Context7 specific enhancements.

        Args:
            query: Original query string
            **kwargs: Additional context including full_code, verbose

        Returns:
            Enhanced query with Context7 optimizations
        """
        full_code = kwargs.get("full_code")
        return self._build_enhanced_query(query, full_code)

    def handle_tool_result(self, result_text: str, tool_name: str, tool_index: int = 1) -> str:
        """Handle tool result with Context7 specific error detection.

        Args:
            result_text: Raw result from tool execution
            tool_name: Name of the executed tool
            tool_index: Index of the tool in current round

        Returns:
            Processed result content
        """
        # Tool result processing
        return self._handle_context7_result(result_text, tool_index)

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
            logger.info("🎯 Timm special handling triggered", tag="context7")

        # Construct enhanced query using experimental template
        enhanced_query = T("rdagent.components.mcp.context7.prompts.templates:context7_enhanced_query_template").r(
            error_message=error_message, context_info=context_info, timm_trigger_text=timm_trigger_text
        )

        return enhanced_query

    # Multi-round calling is now handled by GeneralMCPHandler using LiteLLM backend

    # Tool execution is now handled by GeneralMCPHandler - we provide result processing via handle_tool_result

    def _handle_context7_result(self, result_text: str, tool_index: int) -> str:
        """Handle tool result with Context7 specific error detection."""
        if _is_documentation_not_found(result_text):
            logger.warning(f"📄 Documentation not found", tag="context7")
            return (
                "Documentation not found for this library. This library may not have detailed "
                "documentation available in the Context7 knowledge base, but you can still provide "
                "general guidance based on the library information from resolve-library-id."
            )

        if _is_connection_error(result_text):
            logger.warning(f"🔌 Connection error", tag="context7")
            return "Connection error occurred while fetching documentation. Please try again later."

        return result_text

    # Result logging is now handled by GeneralMCPHandler

    def get_service_info(self) -> dict:
        """Get Context7 service information - simplified."""
        info = super().get_service_info()
        info.update(
            {
                "context7_url": self.mcp_url,
            }
        )
        return info
