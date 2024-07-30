from pathlib import Path

from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelFBWorkspace,
    ModelTask,
)
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class QlibModelExperiment(ModelExperiment[ModelTask, QlibFBWorkspace, ModelFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "model_template")


class QlibModelScenario(Scenario):
    @property
    def background(self) -> str:
        return prompt_dict["qlib_model_background"]

    @property
    def source_data(self) -> str:
        raise NotImplementedError("source_data of QlibModelScenario is not implemented")

    @property
    def output_format(self) -> str:
        return prompt_dict["qlib_model_output_format"]

    @property
    def interface(self) -> str:
        return prompt_dict["qlib_model_interface"]

    @property
    def simulator(self) -> str:
        return prompt_dict["qlib_model_simulator"]

    @property
    def rich_style_description(self)->str:
        return '''
### Qlib Model Evolving Automatic R&D Demo

#### Overview

The demo showcases the iterative process of hypothesis generation, knowledge construction, and decision-making. It highlights how models evolve through continuous feedback and refinement.

#### Key Steps

1. **Hypothesis**
   - Generate and propose initial hypotheses based on data and domain knowledge.

2. **Knowledge Construction**
   - Develop, test, and refine models to gather empirical results.
   - Analyze feedback and incorporate new insights.

3. **Decision Pair**
   - Make data-driven decisions based on validated hypotheses and constructed knowledge.
   - Iterate the process to evolve and improve models continuously.

#### Automated R&D

- **R (Research)**
  - Iteration of ideas and hypotheses.
  - Continuous learning and knowledge construction.

- **D (Development)**
  - Evolving code generation and model refinement.
  - Automated implementation and testing of models.

#### Objective

To demonstrate the dynamic evolution of models through the Qlib platform, emphasizing how each iteration enhances the accuracy and reliability of the resulting models.

        '''

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
