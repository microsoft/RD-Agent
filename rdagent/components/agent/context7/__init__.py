from typing import Optional

from pydantic_ai.mcp import MCPServerStreamableHTTP

from rdagent.components.agent.base import PAIAgent
from rdagent.components.agent.mcp.context7 import SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T


class Agent(PAIAgent):
    """
    A specific agent for context7
    """

    def __init__(self):
        toolsets = [MCPServerStreamableHTTP(SETTINGS.url, timeout=SETTINGS.timeout)]
        super().__init__(system_prompt=T(".prompts:system_prompt").r(), toolsets=toolsets)

    def _build_enhanced_query(self, error_message: str, full_code: Optional[str] = None) -> str:
        """Build enhanced query using experimental prompt templates."""
        # Build context information using template
        context_info = ""
        if full_code:
            context_info = T(".prompts:code_context_template").r(full_code=full_code)

        # Check for timm library special case (experimental optimization)
        timm_trigger = error_message.lower().count("timm") >= 3
        timm_trigger_text = ""
        if timm_trigger:
            timm_trigger_text = T(".prompts:timm_special_case").r()
            logger.info("🎯 Timm special handling triggered", tag="context7")

        # Construct enhanced query using experimental template
        enhanced_query = T(".prompts:context7_enhanced_query_template").r(
            error_message=error_message, context_info=context_info, timm_trigger_text=timm_trigger_text
        )

        return enhanced_query

    def query(self, query: str) -> str:
        """

        Parameters
        ----------
        query : str
            It should be something like error message.

        Returns
        -------
        str
        """
        query = self._build_enhanced_query(error_message=query)
        return super().query(query)
