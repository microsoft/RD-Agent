"""
TODO: Model Structure RD-Loop
TODO: move the following code to a new class: Model_RD_Agent
"""

# import_from

from rdagent.app.qlib_rd_loop.conf import PROP_SETTING
from rdagent.core.developer import Developer
from rdagent.core.exception import ModelEmptyError
from rdagent.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger

scen: Scenario = import_class(PROP_SETTING.model_scen)()

hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.model_hypothesis_gen)(scen)

hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.model_hypothesis2experiment)()

qlib_model_coder: Developer = import_class(PROP_SETTING.model_coder)(scen)
qlib_model_runner: Developer = import_class(PROP_SETTING.model_runner)(scen)

qlib_model_summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.model_summarizer)(scen)

trace = Trace(scen=scen)
with logger.tag("model.loop"):
    for _ in range(PROP_SETTING.evolving_n):
        try:
            with logger.tag("r"):  # research
                hypothesis = hypothesis_gen.gen(trace)
                logger.log_object(hypothesis, tag="hypothesis generation")

                exp = hypothesis2experiment.convert(hypothesis, trace)
                logger.log_object(exp.sub_tasks, tag="experiment generation")
            with logger.tag("d"):  # develop
                exp = qlib_model_coder.develop(exp)
                logger.log_object(exp.sub_workspace_list, tag="model coder result")
            with logger.tag("ef"):  # evaluate and feedback
                exp = qlib_model_runner.develop(exp)
                logger.log_object(exp, tag="model runner result")
                feedback = qlib_model_summarizer.generate_feedback(exp, hypothesis, trace)
                logger.log_object(feedback, tag="feedback")
            trace.hist.append((hypothesis, exp, feedback))
        except ModelEmptyError as e:
            logger.warning(e)
            continue
