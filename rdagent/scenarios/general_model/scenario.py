from copy import deepcopy

from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.utils.agent.tpl import T


class GeneralModelScenario(Scenario):
    def __init__(self) -> None:
        super().__init__()
        self._background = deepcopy(T(".prompts:general_model_background").r())
        self._output_format = deepcopy(T(".prompts:general_model_output_format").r())
        self._interface = deepcopy(T(".prompts:general_model_interface").r())
        self._simulator = deepcopy(T(".prompts:general_model_simulator").r())
        self._rich_style_description = deepcopy(T(".prompts:general_model_rich_style_description").r())

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
