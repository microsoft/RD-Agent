"""
Quant (Factor & Model) workflow with session control
"""

from typing import Any

import fire

from rdagent.app.qlib_rd_loop.conf import QUANT_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import FactorEmptyError
from rdagent.core.exception import ModelEmptyError
from rdagent.log import rdagent_logger as logger
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.core.proposal import (
    Experiment2Feedback,
    Hypothesis2Experiment,
    HypothesisGen,
    Trace,
    HypothesisFeedback,
)
from rdagent.core.developer import Developer

class QuantRDLoop(RDLoop):
    skip_loop_error = (FactorEmptyError,ModelEmptyError, )

    def __init__(self, PROP_SETTING: BasePropSetting):
        with logger.tag("init"):
            scen: Scenario = import_class(PROP_SETTING.scen)()
            logger.log_object(scen, tag="scenario")

            self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.quant_hypothesis_gen)(scen)
            logger.log_object(self.hypothesis_gen, tag="quant hypothesis generator")

            self.factor_hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.factor_hypothesis2experiment)()
            logger.log_object(self.factor_hypothesis2experiment, tag="factor hypothesis2experiment")
            self.model_hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.model_hypothesis2experiment)()
            logger.log_object(self.model_hypothesis2experiment, tag="model hypothesis2experiment")

            self.factor_coder: Developer = import_class(PROP_SETTING.factor_coder)(scen)
            logger.log_object(self.factor_coder, tag="factor coder")
            self.model_coder: Developer = import_class(PROP_SETTING.model_coder)(scen)
            logger.log_object(self.model_coder, tag="model coder")

            self.factor_runner: Developer = import_class(PROP_SETTING.factor_runner)(scen)
            logger.log_object(self.factor_runner, tag="factor runner")
            self.model_runner: Developer = import_class(PROP_SETTING.model_runner)(scen)
            logger.log_object(self.model_runner, tag="model runner")

            self.factor_summarizer: Experiment2Feedback = import_class(PROP_SETTING.factor_summarizer)(scen)
            logger.log_object(self.factor_summarizer, tag="factor summarizer")
            self.model_summarizer: Experiment2Feedback = import_class(PROP_SETTING.model_summarizer)(scen)
            logger.log_object(self.model_summarizer, tag="model summarizer")

            self.trace = Trace(scen=scen)
            super(RDLoop, self).__init__()

    def direct_exp_gen(self, prev_out: dict[str, Any]):
        with logger.tag("r"):  # research
            hypo = self._propose()
            # exp = self._exp_gen(hypo)
            assert hypo.action in ["factor", "model"]
            if hypo.action == "factor":
                exp = self.factor_hypothesis2experiment.convert(hypo, self.trace)
            else:
                exp = self.model_hypothesis2experiment.convert(hypo, self.trace)
            logger.log_object(exp.sub_tasks, tag="experiment generation")
        return {"propose": hypo, "exp_gen": exp}
    
    def coding(self, prev_out: dict[str, Any]):
        with logger.tag("d"):
            if prev_out["direct_exp_gen"]["propose"].action == "factor":
                exp = self.factor_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
            elif prev_out["direct_exp_gen"]["propose"].action == "model":
                exp = self.model_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
            logger.log_object(exp, tag="coder result")
        return exp

    def running(self, prev_out: dict[str, Any]):
        with logger.tag("ef"):
            if prev_out["direct_exp_gen"]["propose"].action == "factor":
                exp = self.factor_runner.develop(prev_out["coding"])
                if exp is None:
                    logger.error(f"Factor extraction failed.")
                    raise FactorEmptyError("Factor extraction failed.")
            elif prev_out["direct_exp_gen"]["propose"].action == "model":
                exp = self.model_runner.develop(prev_out["coding"])
            logger.log_object(exp, tag="runner result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is not None:
            feedback = HypothesisFeedback(
                observations="Error occurred in loop, skip this loop",
                hypothesis_evaluation="",
                new_hypothesis="",
                reason="",
                decision=False,
            )
            self.trace.hist.append((prev_out["direct_exp_gen"]["exp_gen"], feedback))
        else:
            if prev_out["direct_exp_gen"]["propose"].action == "factor":
                feedback = self.factor_summarizer.generate_feedback(prev_out["running"], self.trace)
            elif prev_out["direct_exp_gen"]["propose"].action == "model":
                feedback = self.model_summarizer.generate_feedback(prev_out["running"], self.trace)
            with logger.tag("ef"):
                logger.log_object(feedback, tag="feedback")
            self.trace.hist.append((prev_out["running"], feedback))

def main(path=None, step_n=None):
    """
    Auto R&D Evolving loop for fintech factors.

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/factor.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter

    """
    if path is None:
        quant_loop = QuantRDLoop(QUANT_PROP_SETTING)
    else:
        quant_loop = QuantRDLoop.load(path)
    quant_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)
