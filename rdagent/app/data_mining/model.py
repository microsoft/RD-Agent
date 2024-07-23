"""
TODO: Model Structure RD-Loop
TODO: move the following code to a new class: Model_RD_Agent
"""

# import_from

from rdagent.app.data_mining.conf import PROP_SETTING
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

dm_model_coder: Developer = import_class(PROP_SETTING.model_coder)(scen)
dm_model_runner: Developer = import_class(PROP_SETTING.model_runner)(scen)

dm_model_summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.model_summarizer)(scen)

trace = Trace(scen=scen)
with logger.tag("model.loop"):
    for _ in range(PROP_SETTING.evolving_n):
        try:
            with logger.tag("r"): # research
                hypothesis = hypothesis_gen.gen(trace)
                exp = hypothesis2experiment.convert(hypothesis, trace)
            with logger.tag("d"): # develop
                exp = dm_model_coder.develop(exp)
            with logger.tag("ef"): # evaluate and feedback
                exp = dm_model_runner.develop(exp)
                feedback = dm_model_summarizer.generate_feedback(exp, hypothesis, trace)
            trace.hist.append((hypothesis, exp, feedback))
        except ModelEmptyError as e:
            logger.warning(e)
            continue
