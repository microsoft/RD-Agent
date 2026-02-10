import asyncio
from typing import Any, TYPE_CHECKING

from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import CoderError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.proposal.trace import RLTrace

if TYPE_CHECKING:
    from rdagent.scenarios.rl.scen.scenario import RLPostTrainingScen


class RLPostTrainingRDLoop(RDLoop):
    """RL post-training loop using standard RDLoop workflow"""

    skip_loop_error = (CoderError,)
    skip_loop_error_stepname = "feedback"
    withdraw_loop_error = ()

    def __init__(self, PROP_SETTING: "RLPostTrainingScen"):
        # Store rl-specific settings
        self.rl_rd_setting = PROP_SETTING
        # Initialize using base class
        super().__init__(PROP_SETTING)

        # Replace generic Trace with RLTrace for SOTA tracking
        self.trace = RLTrace(scen=PROP_SETTING)

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        """Generate RL post-training experiment"""
        exp = await self.hypothesis_gen.async_gen(self.trace, self)
        logger.log_object(exp.hypothesis, tag="hypothesis")
        logger.log_object(exp.sub_tasks, tag="experiment generation")
        return exp

    def coding(self, prev_out: dict[str, Any]):
        """Generate rl post-training code"""
        exp = prev_out["direct_exp_gen"]
        exp = self.coder.develop(exp)
        logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        """Generate feedback for RL post-training experiment - always call LLM"""

        # Get experiment from available sources
        exp = prev_out.get("running") or prev_out.get("coding") or prev_out.get("direct_exp_gen")
        e = prev_out.get(self.EXCEPTION_KEY, None)
        feedback = self.summarizer.generate_feedback(exp, self.trace, exception=e)

        logger.log_object(feedback, tag="feedback")
        return feedback

    def record(self, prev_out: dict[str, Any]):
        """Record the experiment and feedback into trace"""
        feedback = prev_out["feedback"]
        exp = prev_out.get("running") or prev_out.get("coding") or prev_out.get("direct_exp_gen")
        self.trace.sync_dag_parent_and_hist((exp, feedback), prev_out[self.LOOP_IDX_KEY])

    def dump(self, path):
        """Skip dump if the loop contains unpicklable objects."""
        try:
            super().dump(path)
        except TypeError as e:
            logger.warning(f"Skip dump due to pickling error: {e}")
