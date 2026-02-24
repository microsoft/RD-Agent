"""
Model workflow with session control
It is from `rdagent/app/qlib_rd_loop/model.py` and try to replace `rdagent/app/qlib_rd_loop/RDAgent.py`
"""

import asyncio
from typing import Any
from multiprocessing import Queue

from rdagent.components.workflow.conf import BasePropSetting
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.proposal import (
    ExperimentPlan,
    Experiment2Feedback,
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisFeedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.utils.workflow import LoopBase, LoopMeta
from rdagent.utils.qlib import ALPHA20, validate_qlib_features


class RDLoop(LoopBase, metaclass=LoopMeta):

    def __init__(self, PROP_SETTING: BasePropSetting):
        scen: Scenario = import_class(PROP_SETTING.scen)()
        logger.log_object(scen, tag="scenario")
        logger.log_object(PROP_SETTING.model_dump(), tag="RDLOOP_SETTINGS")
        logger.log_object(RD_AGENT_SETTINGS.model_dump(), tag="RD_AGENT_SETTINGS")
        self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)

        self.hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()
        self.plan: ExperimentPlan = {"features": ALPHA20} # for user interaction

        self.coder: Developer = import_class(PROP_SETTING.coder)(scen)
        self.runner: Developer = import_class(PROP_SETTING.runner)(scen)

        self.summarizer: Experiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
        self.trace = Trace(scen=scen)
        super().__init__()

    # excluded steps
    def _set_interactor(self, user_request_q: Queue, user_response_q: Queue):
        self.user_request_q = user_request_q
        self.user_response_q = user_response_q
    
    def _interact_init_params(self) -> None:
        if not (hasattr(self, "user_request_q") and hasattr(self, "user_response_q")):
            return

        logger.info("Waiting for user interaction on initial parameters...")
        try:
            self.user_request_q.put({
                "user_instruction": None,
            })
            res_dict = self.user_response_q.get()
            logger.info("Received user instruction response.")
            self.plan.update(res_dict)
            
            fea_valid_msg = ""
            while True:
                logger.info("Requesting base feature configuration from user.")
                self.user_request_q.put({
                    "features": self.plan["features"],
                    "feature_validation_msg": fea_valid_msg,
                })
                self.plan["features"] = self.user_response_q.get()
                logger.info("Received base feature configuration response.")
                if validate_qlib_features(list(self.plan["features"].values())):
                    logger.info(f"Base feature validation passed. {len(self.plan['features'])} features selected.")
                    break
                else:
                    logger.info("Base feature validation failed. Asking user to revise.")
                    fea_valid_msg = "Some features are invalid, please revise."
            
        except (EOFError, OSError):
            logger.info("User interaction failed, using default initial parameters.")
            return
        logger.info("Received user interaction on initial parameters.")

    def _interact_hypo(self, hypo: Hypothesis) -> Hypothesis:
        if not (hasattr(self, "user_request_q") and hasattr(self, "user_response_q")):
            return hypo

        logger.info("Waiting for user interaction on hypothesis...")
        try:
            self.user_request_q.put(hypo.__dict__)
            res_dict = self.user_response_q.get()
            modified_hypo = Hypothesis(**res_dict)
        except (EOFError, OSError):
            logger.info("User interaction failed, using original hypothesis.")
            return hypo
        logger.info("Received user interaction on hypothesis.")
        return modified_hypo

    def _interact_feedback(self, feedback: HypothesisFeedback) -> HypothesisFeedback:
        if not (hasattr(self, "user_request_q") and hasattr(self, "user_response_q")):
            return feedback

        logger.info("Waiting for user interaction on feedback...")
        try:
            self.user_request_q.put(feedback.__dict__)
            res_dict = self.user_response_q.get()
            modified_feedback = HypothesisFeedback(**res_dict)
        except (EOFError, OSError):
            logger.info("User interaction failed, using original feedback.")
            return feedback
        logger.info("Received user interaction on feedback.")
        return modified_feedback

    def _propose(self):
        hypothesis = self.hypothesis_gen.gen(self.trace, self.plan)

        # user can change the hypothesis here
        hypothesis = self._interact_hypo(hypothesis)

        logger.log_object(hypothesis, tag="hypothesis generation")
        return hypothesis

    def _exp_gen(self, hypothesis: Hypothesis):
        exp = self.hypothesis2experiment.convert(hypothesis, self.trace)
        logger.log_object(exp.sub_tasks, tag="experiment generation")
        return exp

    # included steps
    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        while True:
            if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                hypo = self._propose()
                exp = self._exp_gen(hypo)
                exp.base_features = self.plan["features"]
                exp.based_experiments[-1].base_features = self.plan["features"]
                return {"propose": hypo, "exp_gen": exp}
            await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]):
        exp = self.coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    def running(self, prev_out: dict[str, Any]):
        exp = self.runner.develop(prev_out["coding"])
        logger.log_object(exp, tag="runner result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is not None:
            feedback = HypothesisFeedback(
                observations=str(e),
                hypothesis_evaluation="",
                new_hypothesis="",
                reason="",
                decision=False,
                exception=str(e),
            )
            feedback = self._interact_feedback(feedback)
            logger.log_object(feedback, tag="feedback")
            self.trace.hist.append((prev_out["direct_exp_gen"]["exp_gen"], feedback))
        else:
            feedback = self.summarizer.generate_feedback(prev_out["running"], self.trace)
            feedback = self._interact_feedback(feedback)
            logger.log_object(feedback, tag="feedback")
            self.trace.hist.append((prev_out["running"], feedback))

    # TODO: `def record(self, prev_out: dict[str, Any]):` has already been hard coded into LoopBase
    # So we should add it into RDLoop class to make sure every RDLoop Sub Class be aware of it.
