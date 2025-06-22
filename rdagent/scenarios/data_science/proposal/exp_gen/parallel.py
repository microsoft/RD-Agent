from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.proposal import ExpGen
from rdagent.scenarios.data_science.loop import DataScienceRDLoop
from rdagent.scenarios.data_science.proposal.exp_gen.trace_scheduler import RoundRobinScheduler, TraceScheduler

if TYPE_CHECKING:
    from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
    from rdagent.scenarios.data_science.proposal.exp_gen.base import DSTrace
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
        self.exp_gen = DataScienceRDLoop._get_exp_gen(
            "rdagent.scenarios.data_science.proposal.exp_gen.DSExpGen", self.scen
        )
        self.trace_scheduler: TraceScheduler = RoundRobinScheduler()
        self.target_trace_count = DS_RD_SETTING.get("max_traces", 2)

        # # The lock is used to protect the trace context (current_selection)
        # self._trace_context_lock = asyncio.Lock()

    async def async_gen(self, trace: DSTrace, loop: LoopBase) -> DSExperiment:
        """
        Waits for a free execution slot, selects a parent trace using the
        scheduler, generates a new experiment, and injects the parent context
        into it before returning.
        """
        local_selection: tuple[int, ...] = None
        # step 1: select the parant trace to expand
        # Policy: if we have fewer traces than our target, start a new one.
        if trace.sub_trace_count < self.target_trace_count:
            local_selection = trace.NEW_ROOT
        else:
            # Otherwise, use the scheduler to pick an existing trace to expand.
            # NOTE: asyncio.Lock is used in the inner scheduler.
            local_selection = await self.trace_scheduler.select_trace(trace)

        while True:
            if loop.get_unfinished_loop_cnt(loop.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                # step 2: generate the experiment with the local selection
                exp = self.exp_gen.gen(trace, local_selection)

                # Inject the local selection to the experiment object
                exp.set_local_selection(local_selection)

                return exp
            
            await asyncio.sleep(1) 

        