from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.proposal import ExpGen
from rdagent.scenarios.data_science.loop import DataScienceRDLoop
from rdagent.scenarios.data_science.proposal.exp_gen.scheduler import RoundRobinScheduler, TraceScheduler

if TYPE_CHECKING:
    from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
    from rdagent.scenarios.data_science.proposal.exp_gen.base import DSTrace
    from rdagent.utils.workflow.loop import LoopBase


class ParallelExpGen(ExpGen):
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
        self.scheduler: TraceScheduler = RoundRobinScheduler()
        self.target_trace_count = DS_RD_SETTING.get("max_traces", 2)

    async def async_gen(self, trace: DSTrace, loop: LoopBase) -> DSExperiment:
        """
        Waits for a free execution slot, selects a parent trace using the
        scheduler, generates a new experiment, and injects the parent context
        into it before returning.
        """
        parent_selection: tuple[int, ...]

        # Policy: if we have fewer traces than our target, start a new one.
        # This check is not atomic and has a race condition, but for the purpose
        # of this example, it's a simple way to drive trace creation.
        # A more advanced scheduler could manage this internally.
        if trace.sub_trace_count < self.target_trace_count:
            parent_selection = trace.NEW_ROOT
        else:
            # Otherwise, use the scheduler to pick an existing trace to expand.
            parent_selection = await self.scheduler.select_trace(trace)

        # We must set the selection on the trace temporarily for the underlying
        # generator to use it as context. This is a localized change and is
        # safe as long as the underlying gen() is not async.
        trace.set_current_selection(parent_selection)

        # The loop in the base `async_gen` handles the concurrency check.
        # We can call it via super() or reimplement it here.
        while True:
            if loop.get_unfinished_loop_cnt(loop.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                # Generate the base experiment
                exp = self.exp_gen.gen(trace)

                # Inject the context
                exp.parent_selection = parent_selection

                return exp
            
            await asyncio.sleep(1) 