"""
Model workflow with session control
It is from `rdagent/app/qlib_rd_loop/model.py` and try to replace `rdagent/app/qlib_rd_loop/RDAgent.py`
"""

import asyncio
from typing import Any

from rdagent.components.workflow.conf import BasePropSetting
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.proposal import (
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


class RDLoop(LoopBase, metaclass=LoopMeta):

    def __init__(self, PROP_SETTING: BasePropSetting):
        scen: Scenario = import_class(PROP_SETTING.scen)()
        logger.log_object(scen, tag="scenario")
        logger.log_object(PROP_SETTING.model_dump(), tag="RDLOOP_SETTINGS")
        logger.log_object(RD_AGENT_SETTINGS.model_dump(), tag="RD_AGENT_SETTINGS")
        self.hypothesis_gen: HypothesisGen = (
            import_class(PROP_SETTING.hypothesis_gen)(scen)
            if hasattr(PROP_SETTING, "hypothesis_gen") and PROP_SETTING.hypothesis_gen
            else None
        )

        self.hypothesis2experiment: Hypothesis2Experiment = (
            import_class(PROP_SETTING.hypothesis2experiment)()
            if hasattr(PROP_SETTING, "hypothesis2experiment") and PROP_SETTING.hypothesis2experiment
            else None
        )

        self.coder: Developer = (
            import_class(PROP_SETTING.coder)(scen) if hasattr(PROP_SETTING, "coder") and PROP_SETTING.coder else None
        )
        self.runner: Developer = (
            import_class(PROP_SETTING.runner)(scen) if hasattr(PROP_SETTING, "runner") and PROP_SETTING.runner else None
        )

        self.summarizer: Experiment2Feedback = (
            import_class(PROP_SETTING.summarizer)(scen)
            if hasattr(PROP_SETTING, "summarizer") and PROP_SETTING.summarizer
            else None
        )
        self.trace = Trace(scen=scen)
        super().__init__()

    # excluded steps
    def _propose(self):
        hypothesis = self.hypothesis_gen.gen(self.trace)
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
        # TODO: the logic branch of exception should be moved to summarizer
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is not None:
            feedback = HypothesisFeedback(
                reason=str(e),
                decision=False,
                code_change_summary="",
                acceptable=False,
            )
        else:
            feedback = self.summarizer.generate_feedback(prev_out["running"], self.trace)
        logger.log_object(feedback, tag="feedback")
        return feedback

    def record(self, prev_out: dict[str, Any]):
        feedback = prev_out["feedback"]
        exp = prev_out.get("running") or prev_out.get("coding") or prev_out.get("direct_exp_gen", {}).get("exp_gen")
        self.trace.sync_dag_parent_and_hist((exp, feedback), prev_out[self.LOOP_IDX_KEY])
