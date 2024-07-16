from pathlib import Path

from rdagent.components.coder.factor_coder.factor import FactorExperiment
from rdagent.components.coder.factor_coder.utils import get_data_folder_intro
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario

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
    def output_format(self) -> str:
        return prompt_dict["qlib_factor_output_format"]

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
The output of your code should be in the format:
{self.output_format}
The simulator user can use to test your factor:
{self.simulator}
"""
