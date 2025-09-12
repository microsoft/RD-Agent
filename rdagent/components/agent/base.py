from abc import abstractmethod

import nest_asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

from rdagent.oai.backend.pydantic_ai import get_agent_model


class BaseAgent:

    @abstractmethod
    def __init__(self, system_prompt: str, toolsets: list[str]): ...

    @abstractmethod
    def query(self, query: str) -> str: ...


class PAIAgent(BaseAgent):
    """
    Pydantic-AI agent
    """

    agent: Agent

    def __init__(self, system_prompt: str, toolsets: list[str | MCPServerStreamableHTTP]):
        toolsets = [(ts if isinstance(ts, MCPServerStreamableHTTP) else MCPServerStreamableHTTP(ts)) for ts in toolsets]
        self.agent = Agent(get_agent_model(), system_prompt=system_prompt, toolsets=toolsets)

    def query(self, query: str) -> str:
        """

        Parameters
        ----------
        query : str

        Returns
        -------
        str
        """

        nest_asyncio.apply()  # NOTE: very important. Because pydantic-ai uses asyncio!
        result = self.agent.run_sync(query)
        return result.output
