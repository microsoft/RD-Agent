from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.core.scenario import Scenario
from rdagent.components.coder.data_science.workflow.es import (
    WorkflowMultiProcessEvolvingStrategy,
)
from rdagent.components.coder.data_science.workflow.eval import (
    WorkflowGeneralCaseSpecEvaluator,
)


class WorkflowCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        eva = CoSTEERMultiEvaluator(
            WorkflowGeneralCaseSpecEvaluator(scen=scen), scen=scen
        )  # Please specify whether you agree running your eva in parallel or not
        es = WorkflowMultiProcessEvolvingStrategy(scen=scen, settings=CoSTEER_SETTINGS)
        super().__init__(*args, settings=CoSTEER_SETTINGS, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)
