import asyncio
from typing import Any

from rdagent.app.finetune.llm.conf import LLMFinetunePropSetting
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import CoderError
from rdagent.core.proposal import HypothesisFeedback
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.proposal.trace import FTTrace


class LLMFinetuneRDLoop(RDLoop):
    """LLM fine-tuning loop using standard RDLoop workflow"""

    skip_loop_error = (CoderError,)
    withdraw_loop_error = ()

    def __init__(self, PROP_SETTING: LLMFinetunePropSetting):
        # Store finetune-specific settings
        self.ft_rd_setting = PROP_SETTING
        self.dataset = PROP_SETTING.dataset
        self.model = PROP_SETTING.base_model

        # Initialize using base class
        super().__init__(PROP_SETTING)

        # Replace generic Trace with FTTrace for SOTA tracking
        self.trace = FTTrace(scen=self.trace.scen)

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        """Generate LLM fine-tuning experiment"""
        exp = await self.hypothesis_gen.async_gen(self.trace, self)
        logger.log_object(exp.hypothesis, tag="hypothesis")
        logger.log_object(exp.sub_tasks, tag="experiment generation")
        return exp

    def coding(self, prev_out: dict[str, Any]):
        """Generate fine-tuning code"""
        exp = prev_out["direct_exp_gen"]
        exp = self.coder.develop(exp)
        logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        """Generate feedback for LLM fine-tuning experiment - always call LLM"""
        e = prev_out.get(self.EXCEPTION_KEY, None)

        # Get experiment from available sources
        exp = prev_out.get("running") or prev_out.get("coding") or prev_out.get("direct_exp_gen")

        if e is not None:
            # Error case: pass error info to summarizer for LLM analysis
            feedback = self.summarizer.generate_feedback(exp, self.trace, error_info=str(e))
            feedback.acceptable = False
        else:
            # Success case: normal LLM analysis
            feedback = self.summarizer.generate_feedback(exp, self.trace)

        logger.log_object(feedback, tag="feedback")
        return feedback

    def record(self, prev_out: dict[str, Any]):
        """Record the experiment and feedback into trace"""
        feedback = prev_out["feedback"]
        exp = prev_out.get("running") or prev_out.get("coding") or prev_out.get("direct_exp_gen")
        self.trace.sync_dag_parent_and_hist((exp, feedback), prev_out[self.LOOP_IDX_KEY])
