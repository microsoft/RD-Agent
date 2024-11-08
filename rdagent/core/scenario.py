from abc import ABC, abstractmethod

from rdagent.core.experiment import Task


class Scenario(ABC):
    @property
    @abstractmethod
    def background(self) -> str:
        """Background information"""

    # TODO: We have to change all the sub classes to override get_source_data_desc instead of `source_data`
    def get_source_data_desc(self, task: Task | None = None) -> str:  # noqa: ARG002
        """
        Source data description

        The choice of data may vary based on the specific task at hand.
        """
        return ""

    @property
    def source_data(self) -> str:
        """
        A convenient shortcut for describing source data
        """
        return self.get_source_data_desc()

    @property
    @abstractmethod
    def interface(self) -> str:
        """Interface description about how to run the code"""

    @property
    @abstractmethod
    def output_format(self) -> str:
        """Output format description"""

    @property
    @abstractmethod
    def simulator(self) -> str:
        """Simulator description"""

    @property
    @abstractmethod
    def rich_style_description(self) -> str:
        """Rich style description to present"""

    @abstractmethod
    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
    ) -> str:
        """
        Combine all descriptions together

        The scenario description varies based on the task being performed.
        """

    @property
    def experiment_setting(self) -> str | None:
        """Get experiment setting and return as rich text string"""
        return None
