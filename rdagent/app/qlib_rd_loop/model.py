"""
TODO: Model Structure RD-Loop
TODO: move the following code to a new class: Model_RD_Agent
"""

# import_from

from rdagent.app.qlib_rd_loop.conf import PROP_SETTING
from rdagent.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.task_generator import TaskGenerator
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger


scen: Scenario = import_class(PROP_SETTING.qlib_model_scen)()

hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.qlib_model_hypothesis_gen)(scen)

hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.qlib_model_hypothesis2experiment)()

qlib_model_coder: TaskGenerator = import_class(PROP_SETTING.qlib_model_coder)(scen)
qlib_model_runner: TaskGenerator = import_class(PROP_SETTING.qlib_model_runner)(scen)

qlib_model_summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.qlib_model_summarizer)(scen)

trace = Trace(scen=scen)

with logger.tag("model.loop"):
    for _ in range(PROP_SETTING.evolving_n):
        with logger.tag("r"): # research
            hypothesis = hypothesis_gen.gen(trace)
            exp = hypothesis2experiment.convert(hypothesis, trace)
        with logger.tag("d"): # develop
            exp = qlib_model_coder.generate(exp)
        with logger.tag("ef"): # evaluate and feedback
            exp = qlib_model_runner.generate(exp)
            feedback = qlib_model_summarizer.generateFeedback(exp, hypothesis, trace)

        trace.hist.append((hypothesis, exp, feedback))
