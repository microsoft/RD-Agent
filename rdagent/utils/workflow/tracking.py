"""
Tracking module for experiment tracking using MLflow.

This module provides a clean interface for tracking metrics and parameters
while keeping the MLflow dependency optional based on configuration.
"""

import datetime
from typing import TYPE_CHECKING

import pytz

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.log.timer import RD_Agent_TIMER_wrapper

if TYPE_CHECKING:
    # Import here to avoid circular dependency
    from rdagent.utils.workflow.loop import LoopBase

from rdagent.log import rdagent_logger as logger

# Define a placeholder for mlflow if it's not available
mlflow = None

# Conditional import to make MLflow optional
if RD_AGENT_SETTINGS.enable_mlflow:
    try:
        import mlflow  # type: ignore[assignment]
    except ImportError:
        logger.warning("MLflow is enabled in settings but could not be imported.")
        RD_AGENT_SETTINGS.enable_mlflow = False


class WorkflowTracker:
    """
    A workflow-specific tracking system that logs metrics related to workflow execution.

    This class handles metric logging while keeping the MLflow dependency optional.
    If MLflow is not enabled in settings, tracking calls become no-ops.
    """

    def __init__(self, loop_base: "LoopBase"):
        """
        Initialize a WorkflowTracker with a LoopBase instance.

        Args:
            loop_base: The LoopBase instance to track metrics for
        """
        self.loop_base = loop_base

    @staticmethod
    def is_enabled() -> bool:
        """Check if tracking is enabled."""
        return RD_AGENT_SETTINGS.enable_mlflow

    @staticmethod
    def _datetime_to_float(dt: datetime.datetime) -> float:
        """Convert datetime to a structured float representation."""
        return dt.second + dt.minute * 1e2 + dt.hour * 1e4 + dt.day * 1e6 + dt.month * 1e8 + dt.year * 1e10

    def log_workflow_state(self) -> None:
        """
        Log all workflow state metrics from the associated LoopBase instance.
        """
        if not RD_AGENT_SETTINGS.enable_mlflow or mlflow is None:
            return

        # Log workflow progress
        mlflow.log_metric("loop_index", self.loop_base.loop_idx)
        mlflow.log_metric("step_index", self.loop_base.step_idx[self.loop_base.loop_idx])

        current_local_datetime = datetime.datetime.now(pytz.timezone("Asia/Shanghai"))
        float_like_datetime = self._datetime_to_float(current_local_datetime)
        mlflow.log_metric("current_datetime", float_like_datetime)

        # Log API status
        mlflow.log_metric("api_fail_count", RD_Agent_TIMER_wrapper.api_fail_count)
        latest_api_fail_time = RD_Agent_TIMER_wrapper.latest_api_fail_time
        if latest_api_fail_time is not None:
            float_like_datetime = self._datetime_to_float(latest_api_fail_time)
            mlflow.log_metric("lastest_api_fail_time", float_like_datetime)

        # Log timer status if timer is started
        if self.loop_base.timer.started:
            remain_time = self.loop_base.timer.remain_time()
            assert remain_time is not None
            mlflow.log_metric("remain_time", remain_time.seconds)
            mlflow.log_metric(
                "remain_percent",
                remain_time / self.loop_base.timer.all_duration * 100,
            )

    # Keep only the log_workflow_state method as it's the primary entry point now
