"""
TODO: Factor Structure RD-Loop
"""

from dotenv import load_dotenv

from rdagent.core.exception import FactorEmptyException
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger

load_dotenv(override=True)

from rdagent.app.qlib_rd_loop.conf import PROP_SETTING
from rdagent.core.developer import Developer
from rdagent.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.utils import import_class

scen: Scenario = import_class(PROP_SETTING.factor_scen)()

hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.factor_hypothesis_gen)(scen)

hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.factor_hypothesis2experiment)()

qlib_factor_coder: Developer = import_class(PROP_SETTING.factor_coder)(scen)

qlib_factor_runner: Developer = import_class(PROP_SETTING.factor_runner)(scen)

qlib_factor_summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.factor_summarizer)(scen)


trace = Trace(scen=scen)
for _ in range(PROP_SETTING.evolving_n):
    try:
        hypothesis = hypothesis_gen.gen(trace)
        exp = hypothesis2experiment.convert(hypothesis, trace)
        exp = qlib_factor_coder.develop(exp)
        exp = qlib_factor_runner.develop(exp)
        feedback = qlib_factor_summarizer.generateFeedback(exp, hypothesis, trace)

        trace.hist.append((hypothesis, exp, feedback))
    except FactorEmptyException as e:
        logger.warning(e)
        continue
