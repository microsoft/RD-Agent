from pydantic_ai.mcp import MCPServerStreamableHTTP

from rdagent.components.agent.base import PAIAgent
from rdagent.components.agent.mcp.rag import SETTINGS
from rdagent.utils.agent.tpl import T


class Agent(PAIAgent):
    """
    A specific agent for RAG
    """

    def __init__(self):
        toolsets = [MCPServerStreamableHTTP(SETTINGS.url, timeout=SETTINGS.timeout)]
        super().__init__(system_prompt=T(".prompts:system_prompt").r(), toolsets=toolsets)

