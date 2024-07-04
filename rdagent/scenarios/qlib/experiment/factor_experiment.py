from pathlib import Path

from rdagent.components.task_implementation.factor_implementation.factor import (
    FactorExperiment,
)
from rdagent.components.task_implementation.factor_implementation.utils import (
    get_data_folder_intro,
)
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import Scenario

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

QlibFactorExperiment = FactorExperiment


class QlibFactorScenario(Scenario):
    @property
    def background(self) -> str:
        return prompt_dict["qlib_factor_background"]

    @property
    def source_data(self) -> str:
        return get_data_folder_intro()

    @property
    def interface(self) -> str:
        return prompt_dict["qlib_factor_interface"]

    @property
    def simulator(self) -> str:
        return prompt_dict["qlib_factor_simulator"]

    def get_scenario_all_desc(self) -> str:
        return f"""Background of the scenario:
{self.background}
The source data you can use:
{self.source_data}
The interface you should follow to write the runnable code:
{self.interface}
The simulator user can use to test your factor:
{self.simulator}
"""
