from pathlib import Path

from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelFBWorkspace,
    ModelTask,
)
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.scenarios.data_mining.experiment.workspace import DMFBWorkspace

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class DMModelExperiment(ModelExperiment[ModelTask, DMFBWorkspace, ModelFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = DMFBWorkspace(template_folder_path=Path(__file__).parent / "model_template")


class DMModelScenario(Scenario):
    @property
    def background(self) -> str:
        return prompt_dict["dm_model_background"]
    
    @property
    def source_data(self) -> str:
        raise NotImplementedError("source_data is not implemented")
    
    @property
    def output_format(self) -> str:
        return prompt_dict["dm_model_output_format"]

    @property
    def interface(self) -> str:
        return prompt_dict["dm_model_interface"]

    @property
    def simulator(self) -> str:
        return prompt_dict["dm_model_simulator"]
    
    @property
    def rich_style_description(self)->str:
        return "Below is MIMIC Model Evolving Automatic R&D Demo."

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
