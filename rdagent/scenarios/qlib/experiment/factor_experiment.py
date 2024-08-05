from pathlib import Path

from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.utils import get_data_folder_intro
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
        return """
### R&D Agent-Qlib: Automated Quantitative Trading & Iterative Factor Evolution Demo

#### [Overview](#_summary)

The demo showcases the iterative process of hypothesis generation, knowledge construction, and decision-making. It highlights how financial factors evolve through continuous feedback and refinement.

#### Key Steps

1. **Hypothesis Generation**
   - Generate and propose initial hypotheses based on data and domain knowledge.

2. **Factor Creation**
   - Develop, define, and implement new financial factors.
   - Test these factors to gather empirical results.

3. **Factor Validation**
   - Quantitatively validate the newly created factors.

4. **Backtesting with Qlib**
    | **Dataset**      | **Model**   | **Factors**    |
    |------------------|-------------|----------------|
    | 📊 CSI300        | 🤖 LGBModel | 🌟 Alpha158 Plus|


5. **Feedback Analysis**
   - Analyze backtest results.
   - Incorporate feedback to refine hypotheses.

6. **Hypothesis Refinement**
   - Refine hypotheses based on feedback and repeat the process.

#### [Automated R&D](#_rdloops)

- **[R (Research)](#_research)**
  - Iterative development of ideas and hypotheses.
  - Continuous learning and knowledge construction.

- **[D (Development)](#_development)**
  - Progressive implementation and code generation of factors.
  - Automated testing and validation of financial factors.

#### [Objective](#_summary)

To demonstrate the dynamic evolution of financial factors through the Qlib platform, emphasizing how each iteration enhances the accuracy and reliability of the resulting financial factors.

        """

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
