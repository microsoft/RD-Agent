from abc import abstractmethod

import nest_asyncio
from prefect import task
from prefect.cache_policies import INPUTS
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
    Pydantic-AI agent with optional Prefect caching support
    """

    agent: Agent
    enable_cache: bool

    def __init__(
        self,
        system_prompt: str,
        toolsets: list[str | MCPServerStreamableHTTP],
        enable_cache: bool = False,
    ):
        """
        Initialize Pydantic-AI agent

        Parameters
        ----------
        system_prompt : str
            System prompt for the agent
        toolsets : list[str | MCPServerStreamableHTTP]
            List of MCP server URLs or instances
        enable_cache : bool
            Enable persistent caching via Prefect. Requires Prefect server:
            `prefect server start` then set PREFECT_API_URL in environment
        """
        toolsets = [(ts if isinstance(ts, MCPServerStreamableHTTP) else MCPServerStreamableHTTP(ts)) for ts in toolsets]
        self.agent = Agent(get_agent_model(), system_prompt=system_prompt, toolsets=toolsets)
        self.enable_cache = enable_cache

        # Create cached query function if caching is enabled
        if enable_cache:
            self._cached_query = task(cache_policy=INPUTS, persist_result=True)(self._run_query)

    def _run_query(self, query: str) -> str:
        """
        Internal query execution (no caching)
        """
        nest_asyncio.apply()  # NOTE: very important. Because pydantic-ai uses asyncio!
        result = self.agent.run_sync(query)
        return result.output

    def query(self, query: str) -> str:
        """
        Run agent query with optional caching

        Parameters
        ----------
        query : str

        Returns
        -------
        str
        """
        if self.enable_cache:
            return self._cached_query(query)
        else:
            return self._run_query(query)
