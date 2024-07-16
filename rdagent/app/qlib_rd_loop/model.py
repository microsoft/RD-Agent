"""
TODO: Model Structure RD-Loop
"""

from dotenv import load_dotenv

from rdagent.core.scenario import Scenario

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

scen: Scenario = import_class(PROP_SETTING.qlib_model_scen)()

hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.qlib_model_hypothesis_gen)(scen)

hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.qlib_model_hypothesis2experiment)()

qlib_model_coder: Developer = import_class(PROP_SETTING.qlib_model_coder)(scen)
qlib_model_runner: Developer = import_class(PROP_SETTING.qlib_model_runner)(scen)

qlib_model_summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.qlib_model_summarizer)(scen)

trace = Trace(scen=scen)
for _ in range(PROP_SETTING.evolving_n):
    hypothesis = hypothesis_gen.gen(trace)
    exp = hypothesis2experiment.convert(hypothesis, trace)
    exp = qlib_model_coder.develop(exp)
    exp = qlib_model_runner.develop(exp)
    feedback = qlib_model_summarizer.generateFeedback(exp, hypothesis, trace)

    trace.hist.append((hypothesis, exp, feedback))
