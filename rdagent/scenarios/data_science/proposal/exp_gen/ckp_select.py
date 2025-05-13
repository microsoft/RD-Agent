import random

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import CheckpointSelector, Trace
from rdagent.log import rdagent_logger as logger

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
                trace.sub_trace_count += 1
                logger.info(
                    f"SOTA count {sota_count} is below threshold {self.SOTA_COUNT_THRESHOLD}, jump to a new sub-trace"
                )
                logger.info(f"current sub-trace count: {trace.sub_trace_count}")
                return ()
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

                random_choice = random.random()
                if random_choice < 0.5:
                    trace.sub_trace_count += 1
                    logger.info(
                        f"SOTA count {sota_count} is below threshold {self.SOTA_COUNT_THRESHOLD}, jump a new sub-trace"
                    )
                    return ()  # reboot a new sub-trace
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
                        trace.sub_trace_count += 1
                        logger.info(
                            f"SOTA count {sota_count} is below threshold {self.SOTA_COUNT_THRESHOLD}, jump a new sub-trace"
                        )
                        logger.info(f"current sub-trace count: {trace.sub_trace_count}")
                        return ()  # reboot a new sub-trace

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
