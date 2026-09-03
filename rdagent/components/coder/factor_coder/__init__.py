from rdagent.components.coder.factor_coder._costeer_compat import (
    install_compat_redirector,
)

# Re-route the pre-refactor `rdagent.components.coder.factor_coder.CoSTEER.*`
# import path to the new `rdagent.components.coder.CoSTEER.*` package so that
# pre-existing pickled traces (e.g. the `demo_traces` archive referenced in
# the README) continue to deserialise under `rdagent ui`. See #1331.
install_compat_redirector()

from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.evaluators import FactorEvaluatorForCoder
from rdagent.components.coder.factor_coder.evolving_strategy import (
    FactorMultiProcessEvolvingStrategy,
)
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario


class FactorCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        setting = FACTOR_COSTEER_SETTINGS
        eva = CoSTEERMultiEvaluator(FactorEvaluatorForCoder(scen=scen), scen=scen)
        es = FactorMultiProcessEvolvingStrategy(scen=scen, settings=FACTOR_COSTEER_SETTINGS)

        super().__init__(*args, settings=setting, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)

    def develop(self, exp: Experiment) -> Experiment:
        try:
            exp = super().develop(exp)
        finally:
            if hasattr(self, "evolve_agent") and self.evolve_agent.evolving_trace:
                es = self.evolve_agent.evolving_trace[-1]
                exp.prop_dev_feedback = es.feedback
        return exp
