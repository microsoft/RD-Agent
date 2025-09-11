from rdagent.components.agent.mcp.context7 import SETTINGS
from rdagent.components.agent.base import PAIAgent
from rdagent.utils.agent.tpl import T
from pydantic_ai.mcp import MCPServerStreamableHTTP


class Agent(PAIAgent):
    """
    A specific agent for context7
    """

    def __init__(self):
        toolsets = [MCPServerStreamableHTTP(SETTINGS.url, timeout=SETTINGS.timeout)]
        super().__init__(system_prompt=T(".prompts:system_prompt").r(), toolsets=toolsets)

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

        result = self.agent.run_sync(query)
        return result.output
