from pathlib import Path

from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelFBWorkspace,
    ModelTask,
)
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.scenarios.kaggle.experiment.workspace import KGFBWorkspace

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class KGModelExperiment(ModelExperiment[ModelTask, KGFBWorkspace, ModelFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = KGFBWorkspace(template_folder_path=Path(__file__).parent / "model_template")


class KGModelScenario(Scenario):
    @property
    def background(self) -> str:
        return prompt_dict["kg_model_background"]

    @property
    def source_data(self) -> str:
        raise NotImplementedError("source_data is not implemented")

    @property
    def output_format(self) -> str:
        return prompt_dict["kg_model_output_format"]

    @property
    def interface(self) -> str:
        return prompt_dict["kg_model_interface"]

    @property
    def simulator(self) -> str:
        return prompt_dict["kg_model_simulator"]

    @property
    def rich_style_description(self) -> str:
        return """
kaggle scen """

    def get_scenario_all_desc(self) -> str:
        return f"""Background of the scenario:
{self.background}
The interface you should follow to write the runnable code:
{self.interface}
The output of your code should be in the format:
{self.output_format}
The simulator user can use to test your model:
{self.simulator}
"""
