from copy import deepcopy
from pathlib import Path

from rdagent.core.experiment import Task
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class GeneralModelScenario(Scenario):
    def __init__(self) -> None:
        super().__init__()
        self._background = deepcopy(prompt_dict["general_model_background"])
        self._output_format = deepcopy(prompt_dict["general_model_output_format"])
        self._interface = deepcopy(prompt_dict["general_model_interface"])
        self._simulator = deepcopy(prompt_dict["general_model_simulator"])
        self._rich_style_description = deepcopy(prompt_dict["general_model_rich_style_description"])

    @property
    def background(self) -> str:
        return self._background

    @property
    def source_data(self) -> str:
        raise NotImplementedError("source_data of GeneralModelScenario is not implemented")

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def simulator(self) -> str:
        return self._simulator

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:
        return f"""Background of the scenario:
{self.background}
The interface you should follow to write the runnable code:
{self.interface}
The output of your code should be in the format:
{self.output_format}
The simulator user can use to test your model:
{self.simulator}
"""
