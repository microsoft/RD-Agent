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

| Objective         | Description                                                                                             |
|-------------------|---------------------------------------------------------------------------------------------------------|
| **Convenience**   | Provide a tool for financial and quantitative practitioners or enthusiasts to quickly extract and test factors from research reports. |
| **Efficiency**    | Enable rapid identification of factors from a vast number of reports that could enhance the current factor library.                  |
| **Research Facilitation** | Support further research by continuously expanding and refining the factor library.                                      |
| **Innovation**    | Foster innovation in financial analysis by leveraging automated R&D processes to iterate and improve financial factors.             |
        """
