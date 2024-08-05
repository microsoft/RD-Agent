from pathlib import Path

from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
from rdagent.components.coder.factor_coder.utils import get_data_folder_intro
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

#### Key Steps

1. **Hypothesis Generation**
   - Generate and propose initial hypotheses based on insights from financial reports.

2. **Factor Creation**
   - Develop, define, and codify new financial factors derived from the reports.
   - Conduct empirical tests to evaluate these factors.

3. **Factor Validation**
   - Quantitatively validate the newly created factors.

4. **Backtesting with Qlib**
    | **Dataset**      | **Model**   | **Factors**    |
    |------------------|-------------|----------------|
    | 📊 CSI300        | 🤖 LGBModel | 🌟 Alpha158 Plus|

5. **Feedback Analysis**
   - Analyze backtest results.
   - Incorporate feedback to refine and enhance the factor hypotheses.

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
