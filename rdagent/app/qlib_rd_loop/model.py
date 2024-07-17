"""
TODO: Model Structure RD-Loop
TODO: move the following code to a new class: Model_RD_Agent
"""

# import_from

from rdagent.app.qlib_rd_loop.conf import PROP_SETTING
from rdagent.core.developer import Developer
from rdagent.core.exception import ModelEmptyException
from rdagent.core.log import RDAgentLog
from rdagent.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class

scen: Scenario = import_class(PROP_SETTING.model_scen)()

hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.model_hypothesis_gen)(scen)

hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.model_hypothesis2experiment)()

qlib_model_coder: Developer = import_class(PROP_SETTING.model_coder)(scen)
qlib_model_runner: Developer = import_class(PROP_SETTING.model_runner)(scen)

qlib_model_summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.model_summarizer)(scen)

trace = Trace(scen=scen)
for _ in range(PROP_SETTING.evolving_n):
    try:
        hypothesis = hypothesis_gen.gen(trace)
        exp = hypothesis2experiment.convert(hypothesis, trace)
        exp = qlib_model_coder.develop(exp)
        exp = qlib_model_runner.develop(exp)
        feedback = qlib_model_summarizer.generateFeedback(exp, hypothesis, trace)

        trace.hist.append((hypothesis, exp, feedback))
    except ModelEmptyException as e:
        RDAgentLog().warning(e)
        continue
