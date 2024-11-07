from pathlib import Path

from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelFBWorkspace,
    ModelTask,
)
from rdagent.core.experiment import Task
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
    def rich_style_description(self) -> str:
        return """
### MIMIC-III Model Evolving Automatic R&D Demo
 
#### [Overview](#_summary)
 
The demo showcases the iterative process of hypothesis generation, knowledge construction, and decision-making in model construction in a clinical prediction task. The model should predict whether a patient would suffer from Acute Respiratory Failure (ARF) based on first 12 hours ICU monitoring data. 
 
#### [Automated R&D](#_rdloops)
 
- **[R (Research)](#_research)**
  - Iteration of ideas and hypotheses.
  - Continuous learning and knowledge construction.
 
- **[D (Development)](#_development)**
  - Evolving code generation and model refinement.
  - Automated implementation and testing of models.
 
#### [Objective](#_summary)
 
To demonstrate the dynamic evolution of models through the R&D loop, emphasizing how each iteration enhances the model performance and reliability. The performane is measured by the AUROC score (Area Under the Receiver Operating Characteristic), which is a commonly used metric for binary classification.   """

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
