from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.proposal import ExpGen
from rdagent.log import rdagent_logger as logger
from rdagent.log.timer import RD_Agent_TIMER_wrapper, RDAgentTimer
from rdagent.scenarios.data_science.loop import DataScienceRDLoop
from rdagent.scenarios.data_science.proposal.exp_gen.merge import ExpGen2Hypothesis
from rdagent.scenarios.data_science.proposal.exp_gen.trace_scheduler import (
    RoundRobinScheduler,
    TraceScheduler,
)

if TYPE_CHECKING:
    from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
    from rdagent.scenarios.data_science.proposal.exp_gen.base import DSTrace, Experiment
    from rdagent.utils.workflow.loop import LoopBase


class ParallelMultiTraceExpGen(ExpGen):
    """
    An experiment generation strategy that enables parallel multi-trace exploration.

    This generator is designed to work with the "Attribute Injection" model.
    It uses a TraceScheduler to determine which parent node to expand, and
    injects this parent context into the experiment object itself.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The underlying generator for creating a single experiment
        self.exp_gen = DataScienceRDLoop.default_exp_gen(self.scen)
        self.merge_exp_gen = ExpGen2Hypothesis(self.scen)
        self.trace_scheduler: TraceScheduler = RoundRobinScheduler(DS_RD_SETTING.max_trace_num)

    def gen(self, trace: "DSTrace") -> "Experiment":
        raise NotImplementedError(
            "ParallelMultiTraceExpGen is designed for async usage, please call async_gen instead."
        )

    async def async_gen(self, trace: DSTrace, loop: LoopBase) -> DSExperiment:
        """
        Waits for a free execution slot, selects a parent trace using the
        scheduler, generates a new experiment, and injects the parent context
        into it before returning.
        """
        timer: RDAgentTimer = RD_Agent_TIMER_wrapper.timer
        logger.info(f"Remain time: {timer.remain_time()}")
        local_selection: tuple[int, ...] = None

        while True:

            if timer.remain_time() >= timedelta(hours=DS_RD_SETTING.merge_hours):

                if DS_RD_SETTING.enable_inject_knowledge_at_root:

                    if len(trace.hist) == 0:
                        # set the knowledge base option to True for the first trace
                        DS_RD_SETTING.enable_knowledge_base = True

                    else:
                        # set the knowledge base option back to False for the other traces
                        DS_RD_SETTING.enable_knowledge_base = False

                if loop.get_unfinished_loop_cnt(loop.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                    local_selection = await self.trace_scheduler.next(trace)

                    # set the local selection as the global current selection for the trace
                    trace.set_current_selection(local_selection)
                    # step 2: generate the experiment with the local selection
                    exp = self.exp_gen.gen(trace)

                    # Inject the local selection to the experiment object
                    exp.set_local_selection(local_selection)

                    return exp

            else:
                # enter the merging stage
                # make sure the all loops are finished
                if loop.get_unfinished_loop_cnt(loop.loop_idx) < 1:
                    # disable reset in merging stage
                    DS_RD_SETTING.coding_fail_reanalyze_threshold = 100000
                    DS_RD_SETTING.consecutive_errors = 100000

                    leaves: list[int] = trace.get_leaves()
                    if len(leaves) < 2:
                        trace.set_current_selection(selection=(-1,))
                        return self.exp_gen.gen(trace)
                    else:
                        selection = (leaves[0],)
                        if trace.sota_exp_to_submit is not None:
                            for i in range(1, len(leaves)):
                                if trace.is_parent(trace.exp2idx(trace.sota_exp_to_submit), leaves[i]):
                                    selection = (leaves[i],)
                                    break
                        trace.set_current_selection(selection)
                        return self.merge_exp_gen.gen(trace)

            await asyncio.sleep(1)
