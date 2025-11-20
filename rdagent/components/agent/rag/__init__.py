from pydantic_ai.mcp import MCPServerStreamableHTTP

from rdagent.components.agent.base import PAIAgent
from rdagent.components.agent.rag.conf import SETTINGS
from rdagent.utils.agent.tpl import T


class Agent(PAIAgent):
    """
    A specific agent for RAG
    """

    def __init__(self, system_prompt: str | None = None):
        toolsets = [MCPServerStreamableHTTP(SETTINGS.url, timeout=SETTINGS.timeout)]
        if system_prompt is None:
            system_prompt = "You are a Retrieval-Augmented Generation (RAG) agent. Use the retrieved documents to answer the user's queries accurately and concisely."
        super().__init__(system_prompt=system_prompt, toolsets=toolsets)
