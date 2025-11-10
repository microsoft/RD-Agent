
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario


class AgenticSysScen(Scenario):
    def __init__(self, competition: str) -> None:
        self.competition = competition

    # Implement dummy functions for the abstract methods in Scenario
    @property
    def background(self) -> str:
        """Background information"""
        return f"Background for competition: {self.competition}"

    def get_runtime_environment(self) -> str:
        """Get the runtime environment information"""
        return "Runtime environment: Dummy environment for AgenticSysScen"

    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
    ) -> str:
        """Combine all descriptions together"""
        return (
            f"Scenario description for competition: {self.competition}. "
            f"Task: {task}, Filtered tag: {filtered_tag}, Simple background: {simple_background}"
        )

    @property
    def rich_style_description(self) -> str:
        """Rich style description to present"""
        return f"<b>AgenticSysScen</b> for competition: <i>{self.competition}</i>"
