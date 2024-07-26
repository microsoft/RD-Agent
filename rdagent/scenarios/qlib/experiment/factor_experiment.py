from pathlib import Path

from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
from rdagent.components.coder.factor_coder.utils import get_data_folder_intro
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class QlibFactorExperiment(FactorExperiment[FactorTask, QlibFBWorkspace, FactorFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "factor_template")


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

    @property
    def rich_style_description(self) -> str:
        return "Below is QlibFactor Evolving Automatic R&D Demo."

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
