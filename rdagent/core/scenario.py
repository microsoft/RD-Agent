from abc import ABC, abstractmethod


class Scenario(ABC):
    @property
    @abstractmethod
    def background(self) -> str:
        """Background information"""

    @property
    @abstractmethod
    def source_data(self) -> str:
        """Source data description"""

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
    def get_scenario_all_desc(self) -> str:
        """Combine all the description together"""



    
