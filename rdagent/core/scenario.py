from abc import ABC, abstractmethod


class Scenario(ABC):
    @property
    @abstractmethod
    def background(self):
        """Background information"""

    @property
    @abstractmethod
    def source_data(self):
        """Source data description"""

    @property
    @abstractmethod
    def interface(self):
        """Interface description about how to run the code"""

    @property
    @abstractmethod
    def output_format(self):
        """Output format description"""

    @property
    @abstractmethod
    def simulator(self):
        """Simulator description"""

    @abstractmethod
    def get_scenario_all_desc(self) -> str:
        """Combine all the description together"""
