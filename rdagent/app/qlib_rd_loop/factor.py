"""
Factor Structure RD-Loop
"""

from dotenv import load_dotenv

from rdagent.core.exception import FactorEmptyError
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
        with logger.tag("r"):  # research
            hypothesis = hypothesis_gen.gen(trace)
            logger.log_object(hypothesis, tag="hypothesis generation")

            exp = hypothesis2experiment.convert(hypothesis, trace)
            logger.log_object(exp.sub_tasks, tag="experiment generation")

        with logger.tag("d"):
            exp = qlib_factor_coder.develop(exp)
            logger.log_object(exp.sub_workspace_list)

        with logger.tag("ef"):
            exp = qlib_factor_runner.develop(exp)
            logger.log_object(exp, tag="factor runner result")
            feedback = qlib_factor_summarizer.generate_feedback(exp, hypothesis, trace)
            logger.log_object(feedback, tag="feedback")

        trace.hist.append((hypothesis, exp, feedback))
    except FactorEmptyError as e:
        logger.warning(e)
        continue
