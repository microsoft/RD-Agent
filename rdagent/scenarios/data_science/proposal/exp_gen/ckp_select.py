import random
from datetime import datetime, timedelta

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import CheckpointSelector, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.log.timer import RD_Agent_TIMER_wrapper, RDAgentTimer

# # TODO: more advanced selector
# # TODO/Discussion: load selector function here or define selector class in `proposal.py`?


class LatestCKPSelector(CheckpointSelector):
    """
    -`(-1, )` represents starting from the latest trial in the trace
    """

    def __init__(
        self,
    ):
        logger.info(f"Using latest selector by default")

    def get_selection(self, trace: Trace) -> tuple[int, ...]:

        return (-1,)


class LimitTimeCKPSelector(CheckpointSelector):
    """
    recore the time of current sub-trace, and jump to a new sub-trace if the time is up
    """

    def __init__(
        self,
    ):
        self.global_timer: RDAgentTimer = RD_Agent_TIMER_wrapper.timer
        self.sub_trace_start_times = {}
        self.MAX_TRACE_NUM = DS_RD_SETTING.max_trace_num
        self.time_limit_pre_trace = None

    def set_time_limit(self):

        # Calculate total time excluding merge hours
        remaining_time = (
            self.global_timer.all_duration.total_seconds() - timedelta(hours=DS_RD_SETTING.merge_hours).total_seconds()
        )
        # Convert to timedelta after division
        self.time_limit_pre_trace = timedelta(seconds=remaining_time / DS_RD_SETTING.max_trace_num)
        # Track when each sub-trace starts
        logger.info(f"Using limit time selector with time limit {self.time_limit_pre_trace} per trace")

    def get_selection(self, trace: Trace) -> tuple[int, ...]:
        """
        Determine whether to continue with the current sub-trace or start a new one
        based on the time spent in the current sub-trace.

        Returns:
            (-1,): Continue with the current latest trial
            trace.NEW_ROOT: Start a new sub-trace if max trace limit not reached
        """

        if self.time_limit_pre_trace is None:
            self.set_time_limit()

        current_time = datetime.now()

        if len(trace.hist) == 0:
            self.sub_trace_start_times[trace.sub_trace_count] = current_time
            logger.info(f"Starting initial sub-trace {trace.sub_trace_count} at {current_time}")
            return (-1,)  # Continue with latest trial for new sub-trace

        # Calculate elapsed time for current sub-trace, Trace count may be larger than MAX_TRACE_NUM druing merge process
        elapsed_time = current_time - self.sub_trace_start_times[min(trace.sub_trace_count, self.MAX_TRACE_NUM) - 1]

        if elapsed_time < self.time_limit_pre_trace:
            # Continue with current sub-trace
            logger.info(
                f"Elapsed time {elapsed_time} is below time limit {self.time_limit_pre_trace}, continue the current sub-trace"
            )
            logger.info(f"current sub-trace count: {trace.sub_trace_count}")
            return (-1,)
        else:
            # Check if we've reached the maximum number of traces
            if trace.sub_trace_count >= self.MAX_TRACE_NUM:
                logger.info(
                    f"Reached maximum trace count ({self.MAX_TRACE_NUM}), continuing with the current sub-trace"
                )
                logger.info(f"current sub-trace count: {trace.sub_trace_count}")
                return (-1,)

            # Time limit exceeded, start a new sub-trace
            self.sub_trace_start_times[trace.sub_trace_count] = current_time
            logger.info(
                f"Elapsed time {elapsed_time} exceeds time limit {self.time_limit_pre_trace}, jump to a new sub-trace"
            )
            logger.info(f"current sub-trace count: {trace.sub_trace_count}")
            return trace.NEW_ROOT  # Empty tuple signals starting a new sub-trace


class SOTAJumpCKPSelector(CheckpointSelector):
    """
    SOTA jump policy:
    if the cumulative SOTA in a window is below a threshold, jump to a new trial
    otherwise, continue the current latest trial
    """

    def __init__(
        self,
    ) -> None:
        self.SOTA_COUNT_WINDOW = DS_RD_SETTING.sota_count_window
        self.SOTA_COUNT_THRESHOLD = DS_RD_SETTING.sota_count_threshold
        self.MAX_TRACE_NUM = DS_RD_SETTING.max_trace_num

        logger.info(
            f"Using SOTA-jump selector with window {self.SOTA_COUNT_WINDOW} and threshold {self.SOTA_COUNT_THRESHOLD}"
        )

    def get_selection(self, trace: Trace) -> tuple[int, ...]:
        current_trace = trace.retrieve_search_list(search_type="ancestors")
        if len(trace.hist) > 0 and len(current_trace) > self.SOTA_COUNT_WINDOW:
            all_exp_list = trace.experiment_and_feedback_list_after_init(return_type="all", search_type="ancestors")
            # sota_exp_list = trace.experiment_and_feedback_list_after_init(return_type="sota", search_type="ancestors")
            exp_list_in_window = all_exp_list[-self.SOTA_COUNT_WINDOW :]

            # compute the cumulative SOTA ratio in the window
            sota_count = 0
            for exp, fb in exp_list_in_window:
                if fb.decision:
                    sota_count += 1
            if sota_count < self.SOTA_COUNT_THRESHOLD:
                # Check if we've reached the maximum number of traces
                if trace.sub_trace_count >= self.MAX_TRACE_NUM:
                    logger.info(
                        f"Reached maximum trace count ({self.MAX_TRACE_NUM}), continuing with the current sub-trace"
                    )
                    logger.info(f"current sub-trace count: {trace.sub_trace_count}")
                    return (-1,)

                logger.info(
                    f"SOTA count {sota_count} is below threshold {self.SOTA_COUNT_THRESHOLD}, jump to a new sub-trace"
                )
                logger.info(f"current sub-trace count: {trace.sub_trace_count}")
                return trace.NEW_ROOT
            else:
                logger.info(
                    f"SOTA count {sota_count} is above threshold {self.SOTA_COUNT_THRESHOLD}, continue the current latest trial"
                )
                logger.info(f"current sub-trace count: {trace.sub_trace_count}")
                return (-1,)

        else:
            logger.info(f"Not enough history to make a decision, continue the current latest trial")
            return (-1,)


class BackJumpCKPSelector(CheckpointSelector):
    """
    back-jump policy:
    if the cumulative SOTA in a window is below a threshold,
    with 50% probability, reboot a new sub-trace
    with 50% probability, jump back to the "last second" SOTA trial (we assume the lastest SOTA trial is not good enough selection)
    """

    def __init__(
        self,
    ) -> None:
        self.SOTA_COUNT_WINDOW = DS_RD_SETTING.sota_count_window
        self.SOTA_COUNT_THRESHOLD = DS_RD_SETTING.sota_count_threshold
        self.MAX_TRACE_NUM = DS_RD_SETTING.max_trace_num

        logger.info(
            f"Using back-jump selector with window {self.SOTA_COUNT_WINDOW} and threshold {self.SOTA_COUNT_THRESHOLD}"
        )

    def get_selection(self, trace: Trace) -> tuple[int, ...]:
        current_trace = trace.retrieve_search_list(search_type="ancestors")

        if len(trace.hist) > 0 and len(current_trace) > self.SOTA_COUNT_WINDOW:

            all_exp_list = trace.experiment_and_feedback_list_after_init(return_type="all", search_type="ancestors")
            # sota_exp_list = trace.experiment_and_feedback_list_after_init(return_type="sota", search_type="ancestors")
            exp_list_in_window = all_exp_list[-self.SOTA_COUNT_WINDOW :]

            # compute the cumulative SOTA ratio in the window
            sota_count = 0
            for exp, fb in exp_list_in_window:
                if fb.decision:
                    sota_count += 1

            if sota_count < self.SOTA_COUNT_THRESHOLD:
                # Check if we've reached the maximum number of traces before creating a new one
                if trace.sub_trace_count >= self.MAX_TRACE_NUM:
                    logger.info(
                        f"Reached maximum trace count ({self.MAX_TRACE_NUM}), continuing with the current sub-trace"
                    )
                    logger.info(f"current sub-trace count: {trace.sub_trace_count}")
                    return (-1,)

                random_choice = random.random()
                if random_choice < 0.5:
                    logger.info(
                        f"SOTA count {sota_count} is below threshold {self.SOTA_COUNT_THRESHOLD}, jump a new sub-trace"
                    )
                    return trace.NEW_ROOT  # reboot a new sub-trace
                else:
                    logger.info(
                        f"SOTA count {sota_count} is below threshold {self.SOTA_COUNT_THRESHOLD}, jump back to the last second SOTA in hist (may not in current sub-trace)"
                    )
                    sota_exp_list = trace.experiment_and_feedback_list_after_init(return_type="sota", search_type="all")
                    if len(sota_exp_list) > 1:
                        last_second_sota_idx = trace.hist.index(sota_exp_list[-2])
                        logger.info(
                            f"jump back to the last second SOTA in hist (may not in current sub-trace), index: {last_second_sota_idx}"
                        )
                        logger.info(f"current sub-trace count: {trace.sub_trace_count}")
                        return (last_second_sota_idx,)
                    else:
                        # Check max trace limit again before creating a new trace
                        if trace.sub_trace_count >= self.MAX_TRACE_NUM:
                            logger.info(
                                f"Reached maximum trace count ({self.MAX_TRACE_NUM}), continuing with the current sub-trace"
                            )
                            logger.info(f"current sub-trace count: {trace.sub_trace_count}")
                            return (-1,)

                        logger.info(
                            f"SOTA count {sota_count} is below threshold {self.SOTA_COUNT_THRESHOLD}, jump a new sub-trace"
                        )
                        logger.info(f"current sub-trace count: {trace.sub_trace_count}")
                        return trace.NEW_ROOT  # reboot a new sub-trace

            else:
                logger.info(
                    f"SOTA count {sota_count} is above threshold {self.SOTA_COUNT_THRESHOLD}, continue the current latest trial"
                )
                logger.info(f"current sub-trace count: {trace.sub_trace_count}")
                return (-1,)
        else:
            logger.info(f"Not enough history to make a decision, continue the current latest trial")
            logger.info(f"current sub-trace count: {trace.sub_trace_count}")
            return (-1,)


# TODO: implement these selectors and more
