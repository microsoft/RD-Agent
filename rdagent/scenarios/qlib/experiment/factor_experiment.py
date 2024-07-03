from pathlib import Path
from rdagent.components.task_implementation.factor_implementation.factor import (
    FactorExperiment,
)
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import Scenario

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

QlibFactorExperiment = FactorExperiment

class QlibFactorScenario(Scenario):
    