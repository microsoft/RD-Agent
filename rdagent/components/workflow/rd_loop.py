"""
Model workflow with session control
It is from `rdagent/app/qlib_rd_loop/model.py` and try to replace `rdagent/app/qlib_rd_loop/RDAgent.py`
"""

import time
from typing import Any

from rdagent.components.workflow.conf import BasePropSetting
from rdagent.core.developer import Developer
from rdagent.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.log.time import measure_time
from rdagent.utils.workflow import LoopBase, LoopMeta


class RDLoop(LoopBase, metaclass=LoopMeta):
    @measure_time
    def __init__(self, PROP_SETTING: BasePropSetting):
        with logger.tag("init"):
            scen: Scenario = import_class(PROP_SETTING.scen)()
            logger.log_object(scen, tag="scenario")

            self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)
            logger.log_object(self.hypothesis_gen, tag="hypothesis generator")

            self.hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()
            logger.log_object(self.hypothesis2experiment, tag="hypothesis2experiment")

            self.coder: Developer = import_class(PROP_SETTING.coder)(scen)
            logger.log_object(self.coder, tag="coder")
            self.runner: Developer = import_class(PROP_SETTING.runner)(scen)
            logger.log_object(self.runner, tag="runner")

            self.summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
            logger.log_object(self.summarizer, tag="summarizer")
            self.trace = Trace(scen=scen)
            super().__init__()

    @measure_time
    def propose(self, prev_out: dict[str, Any]):
        with logger.tag("r"):  # research
            hypothesis = self.hypothesis_gen.gen(self.trace)
            logger.log_object(hypothesis, tag="hypothesis generation")
        return hypothesis

    @measure_time
    def exp_gen(self, prev_out: dict[str, Any]):
        with logger.tag("r"):  # research
            exp = self.hypothesis2experiment.convert(prev_out["propose"], self.trace)
            logger.log_object(exp.sub_tasks, tag="experiment generation")
        return exp

    @measure_time
    def coding(self, prev_out: dict[str, Any]):
        with logger.tag("d"):  # develop
            exp = self.coder.develop(prev_out["exp_gen"])
            logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    @measure_time
    def running(self, prev_out: dict[str, Any]):
        with logger.tag("ef"):  # evaluate and feedback
            exp = self.runner.develop(prev_out["coding"])
            logger.log_object(exp, tag="runner result")
        return exp

    @measure_time
    def feedback(self, prev_out: dict[str, Any]):
        feedback = self.summarizer.generate_feedback(prev_out["running"], prev_out["propose"], self.trace)
        with logger.tag("ef"):  # evaluate and feedback
            logger.log_object(feedback, tag="feedback")
        self.trace.hist.append((prev_out["propose"], prev_out["running"], feedback))
