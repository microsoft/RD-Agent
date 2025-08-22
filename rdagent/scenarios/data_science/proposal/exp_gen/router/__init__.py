from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.proposal import ExpGen, ExpPlanner
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.log.timer import RD_Agent_TIMER_wrapper, RDAgentTimer
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.loop import DataScienceRDLoop
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.draft.draft import DSDraftV2ExpGen
from rdagent.scenarios.data_science.proposal.exp_gen.merge import ExpGen2Hypothesis
from rdagent.scenarios.data_science.proposal.exp_gen.planner import (
    DSExperimentPlan,
    ExperimentPlan,
)
from rdagent.scenarios.data_science.proposal.exp_gen.proposal import DSProposalV2ExpGen
from rdagent.scenarios.data_science.proposal.exp_gen.trace_scheduler import (
    RoundRobinScheduler,
    SOTABasedScheduler,
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
        self.draft_exp_gen = DSDraftV2ExpGen(self.scen)
        self.merge_exp_gen = ExpGen2Hypothesis(self.scen)
        # self.trace_scheduler: TraceScheduler = RoundRobinScheduler(DS_RD_SETTING.max_trace_num)
        self.trace_scheduler: TraceScheduler = import_class(DS_RD_SETTING.trace_scheduler)(
            DS_RD_SETTING.max_trace_num,
            DS_RD_SETTING.scheduler_temperature,
        )
        self.planner = import_class(DS_RD_SETTING.planner)(self.scen)

    def gen(
        self,
        trace: "DSTrace",
        plan: "ExperimentPlan" | None = None,
    ) -> "Experiment":
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
            if loop.get_unfinished_loop_cnt(loop.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                # set trace current selection
                leaves: list[int] = trace.get_leaves()
                if not timer.started or timer.remain_time() >= timedelta(hours=DS_RD_SETTING.merge_hours):
                    local_selection = await self.trace_scheduler.next(trace)

                    # set the local selection as the global current selection for the trace
                    trace.set_current_selection(local_selection)
                else:
                    if len(leaves) < 2:
                        local_selection = (-1,)
                        trace.set_current_selection(selection=local_selection)
                    else:
                        local_selection = (leaves[0],)
                        if trace.sota_exp_to_submit is not None:
                            for i in range(1, len(leaves)):
                                if trace.is_parent(trace.exp2idx(trace.sota_exp_to_submit), leaves[i]):
                                    local_selection = (leaves[i],)
                                    break
                        trace.set_current_selection(local_selection)

                ds_plan = self.planner.plan(trace) if DS_RD_SETTING.enable_planner else DSExperimentPlan()

                start = datetime.now(timezone.utc)
                exp_gen_type = ""
                if (
                    (not timer.started or timer.remain_time() >= timedelta(hours=DS_RD_SETTING.merge_hours))
                    and trace.sota_experiment(selection=local_selection) is None
                    and DS_RD_SETTING.enable_draft_before_first_sota
                ):
                    exp = self.draft_exp_gen.gen(trace, plan=ds_plan)
                    exp_gen_type = type(self.draft_exp_gen).__name__
                elif (
                    timer.started
                    and timer.remain_time() < timedelta(hours=DS_RD_SETTING.merge_hours)
                    and len(leaves) >= 2
                ):
                    DS_RD_SETTING.coding_fail_reanalyze_threshold = 100000
                    DS_RD_SETTING.consecutive_errors = 100000
                    exp = self.merge_exp_gen.gen(trace, plan=ds_plan)
                    exp_gen_type = type(self.merge_exp_gen).__name__
                else:
                    # If there is a sota experiment in the sub-trace and not in merge time, we use default exp_gen
                    exp = self.exp_gen.gen(trace, plan=ds_plan)
                    exp_gen_type = type(self.exp_gen).__name__
                end = datetime.now(timezone.utc)
                logger.log_object(
                    {
                        "exp_gen_type": exp_gen_type,
                        "start_time": start,
                        "end_time": end,
                    },
                    tag="exp_gen_time_info",
                )
                exp.set_local_selection(local_selection)
                exp.plan = ds_plan

                # Register the newly created experiment before returning
                trace.register_uncommitted_exp(exp, loop.loop_idx)
                return exp

            await asyncio.sleep(1)
