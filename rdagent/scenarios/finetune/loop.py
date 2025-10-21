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
        exp = await self.hypothesis_gen.async_gen(self.trace, self)
        logger.log_object(exp.sub_tasks, tag="experiment generation")
        return exp

    def coding(self, prev_out: dict[str, Any]):
        """Generate fine-tuning code"""
        exp = prev_out["direct_exp_gen"]

        # TODO: Single-stage execution (future: multi-stage like DataScience)
        # Currently, FT always has exactly one task stage per experiment.
        # Unlike DataScience which iterates: Feature -> Model -> Ensemble,
        # FT processes one TrainingTask per loop and relies on loop iterations for exploration.
        #
        # This pattern is kept for future multi-stage scenarios:
        #   - Stage 1: Data augmentation/processing
        #   - Stage 2: Supervised fine-tuning
        #   - Stage 3: RLHF/reward modeling
        #
        # When implementing multi-stage:
        #   - Add loop: for tasks in exp.pending_tasks_list:
        #   - Add stage-specific coder selection based on task type
        #   - Handle inter-stage data passing
        if hasattr(exp, "pending_tasks_list") and exp.pending_tasks_list:
            exp.sub_tasks = exp.pending_tasks_list[0]  # Always single stage for now

        exp = self.coder.develop(exp)
        logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp
