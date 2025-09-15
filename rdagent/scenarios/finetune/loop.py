import asyncio
from typing import Any

from rdagent.app.finetune.llm.conf import LLMFinetunePropSetting
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger


class LLMFinetuneRDLoop(RDLoop):
    """LLM fine-tuning loop using standard RDLoop workflow"""

    def __init__(self, PROP_SETTING: LLMFinetunePropSetting):
        # Store finetune-specific settings
        self.ft_rd_setting = PROP_SETTING
        self.dataset = PROP_SETTING.dataset
        self.model = PROP_SETTING.base_model

        # Initialize using base class
        super().__init__(PROP_SETTING)

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        """Generate LLM fine-tuning experiment"""
        if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
            exp = self.hypothesis_gen.gen(self.trace)
            logger.log_object(exp.sub_tasks, tag="experiment generation")
            return exp

        await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]):
        """Generate fine-tuning code"""
        exp = prev_out["direct_exp_gen"]

        # Handle pending tasks like in data science loop
        if hasattr(exp, "pending_tasks_list") and exp.pending_tasks_list:
            exp.sub_tasks = exp.pending_tasks_list[0]  # For finetune, typically one task group

        exp = self.coder.develop(exp)
        logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    def running(self, prev_out: dict[str, Any]):
        """Execute fine-tuning experiment"""
        exp = prev_out["coding"]
        exp = self.runner.develop(exp)
        logger.log_object(exp, tag="runner result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        """Generate feedback from experiment results (inherits base behavior)"""
        # Use base class implementation - it's already optimal
        super().feedback(prev_out)
