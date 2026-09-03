import asyncio
from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.merge import (
    ExpGen2TraceAndMerge,
    ExpGen2TraceAndMergeV2,
    ExpGen2TraceAndMergeV3,
)
from rdagent.scenarios.data_science.proposal.exp_gen.router import ParallelMultiTraceExpGen


def make_experiment() -> DSExperiment:
    return DSExperiment.__new__(DSExperiment)


@pytest.mark.offline
@pytest.mark.parametrize("generator_class", [ExpGen2TraceAndMerge, ExpGen2TraceAndMergeV2, ExpGen2TraceAndMergeV3])
def test_merge_generators_use_per_experiment_state(generator_class: type) -> None:
    experiment = make_experiment()
    merge_exp_gen = Mock()
    merge_exp_gen.gen.return_value = experiment
    generator = object.__new__(generator_class)
    generator.exp_gen = Mock()
    generator.merge_exp_gen = merge_exp_gen
    if generator_class is ExpGen2TraceAndMergeV2:
        generator.flag_start_merge = False

    trace = Mock()
    trace.NEW_ROOT = ()
    trace.sota_exp_to_submit = None
    trace.get_leaves.return_value = [0, 1]
    trace.sub_trace_count = 2

    original_thresholds = (
        DS_RD_SETTING.coding_fail_reanalyze_threshold,
        DS_RD_SETTING.consecutive_errors,
    )
    with patch(
        "rdagent.scenarios.data_science.proposal.exp_gen.merge.RD_Agent_TIMER_wrapper.timer",
    ) as timer:
        timer.remain_time.return_value = timedelta()
        result = generator.gen(trace)

    assert result.is_merge_phase is True
    assert (
        DS_RD_SETTING.coding_fail_reanalyze_threshold,
        DS_RD_SETTING.consecutive_errors,
    ) == original_thresholds


@pytest.mark.offline
def test_parallel_merge_generator_uses_per_experiment_state() -> None:
    experiment = make_experiment()
    generator = object.__new__(ParallelMultiTraceExpGen)
    generator.merge_exp_gen = Mock()
    generator.merge_exp_gen.gen.return_value = experiment
    generator.exp_gen = Mock()
    generator.draft_exp_gen = Mock()
    generator.planner = Mock()
    generator.trace_scheduler = Mock()

    trace = Mock()
    trace.get_leaves.return_value = [0, 1]
    trace.sota_exp_to_submit = None
    loop = Mock()
    loop.loop_idx = 1
    loop.get_unfinished_loop_cnt.return_value = 0

    original_thresholds = (
        DS_RD_SETTING.coding_fail_reanalyze_threshold,
        DS_RD_SETTING.consecutive_errors,
    )
    with (
        patch(
            "rdagent.scenarios.data_science.proposal.exp_gen.router.RD_Agent_TIMER_wrapper.timer",
        ) as timer,
        patch.object(DS_RD_SETTING, "enable_planner", new=False),
    ):
        timer.started = True
        timer.remain_time.return_value = timedelta()
        result = asyncio.run(generator.async_gen(trace, loop))

    assert result.is_merge_phase is True
    assert (
        DS_RD_SETTING.coding_fail_reanalyze_threshold,
        DS_RD_SETTING.consecutive_errors,
    ) == original_thresholds


@pytest.mark.offline
def test_regular_experiment_enables_auto_recovery_by_default() -> None:
    assert make_experiment().is_merge_phase is False
