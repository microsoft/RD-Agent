"""
Model workflow with session control
It is from `rdagent/app/qlib_rd_loop/model.py` and try to replace `rdagent/app/qlib_rd_loop/RDAgent.py`
"""

import fire
from typing import Any
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

from rdagent.utils.workflow import LoopMeta, LoopBase

class ModelLoop(LoopBase, metaclass=LoopMeta):
    # TODO: supporting customized loop control like catching `ModelEmptyError`

    def __init__(self):
        scen: Scenario = import_class(PROP_SETTING.model_scen)()

        self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.model_hypothesis_gen)(scen)

        self.hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.model_hypothesis2experiment)()

        self.qlib_model_coder: Developer = import_class(PROP_SETTING.model_coder)(scen)
        self.qlib_model_runner: Developer = import_class(PROP_SETTING.model_runner)(scen)

        self.qlib_model_summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.model_summarizer)(scen)
        self.trace = Trace(scen=scen)
        super().__init__()

    def propose(self, prev_out: dict[str, Any]):
        with logger.tag("r"):  # research
            hypothesis = self.hypothesis_gen.gen(self.trace)
            logger.log_object(hypothesis, tag="hypothesis generation")
        return hypothesis

    def exp_gen(self, prev_out: dict[str, Any]):
        with logger.tag("r"):  # research
            exp = self.hypothesis2experiment.convert(prev_out["propose"], self.trace)
            logger.log_object(exp.sub_tasks, tag="experiment generation")
        return exp

    def coding(self, prev_out: dict[str, Any]):
        with logger.tag("d"):  # develop
            exp = self.qlib_model_coder.develop(prev_out["exp_gen"])
            logger.log_object(exp.sub_workspace_list, tag="model coder result")
        return exp

    def running(self, prev_out: dict[str, Any]):
        with logger.tag("ef"):  # evaluate and feedback
            exp = self.qlib_model_runner.develop(prev_out["coding"])
            logger.log_object(exp, tag="model runner result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        feedback = self.qlib_model_summarizer.generate_feedback(prev_out["running"], prev_out["propose"], self.trace)
        logger.log_object(feedback, tag="feedback")
        self.trace.hist.append((prev_out["propose"],prev_out["running"] , feedback))


def main(path=None, step_n=None):
    """
    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/model_w_sc.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter

    """
    if path is None:
        model_loop = ModelLoop()
    else:
        model_loop = ModelLoop.load(path)
    model_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)
