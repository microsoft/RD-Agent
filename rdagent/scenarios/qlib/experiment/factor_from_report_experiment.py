from pathlib import Path

from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class QlibFactorFromReportScenario(QlibFactorScenario):
    @property
    def rich_style_description(self) -> str:
        return """
### R&D Agent-Qlib: Automated Quantitative Trading & Factor Extraction from Financial Reports Demo


#### [Overview](#_summary)

This demo showcases the process of extracting factors from financial research reports, implementing these factors, and analyzing their performance through Qlib backtesting, continually expanding and refining the factor library.

#### [Automated R&D](#_rdloops)

- **[R (Research)](#_research)**
  - Iterative development of ideas and hypotheses from financial reports.
  - Continuous learning and knowledge construction.

- **[D (Development)](#_development)**
  - Progressive factor extraction and code generation.
  - Automated implementation and testing of financial factors.

#### [Objective](#_summary)

<table border="1" style="width:100%; border-collapse: collapse;">
  <tr>
    <td>💡 <strong>Innovation </strong></td>
    <td>Tool to quickly extract and test factors from research reports.</td>
  </tr>
  <tr>
    <td>⚡ <strong>Efficiency </strong></td>
    <td>Rapid identification of valuable factors from numerous reports.</td>
  </tr>
  <tr>
    <td>🗃️ <strong>Outputs </strong></td>
    <td>Expand and refine the factor library to support further research.</td>
  </tr>
</table>
        """

    @property
    def get_experiment_setting(self) -> str:
        return """
| Dataset 📊 | Model 🤖    | Factors 🌟       | Data Split  🧮                                   |
|---------|----------|---------------|-------------------------------------------------|
| CSI300  | LGBModel | Alpha158 Plus | Train: 2008-01-01 to 2014-12-31 <br> Valid: 2015-01-01 to 2016-12-31 <br> Test &nbsp;: 2017-01-01 to 2020-08-01 |
        """
