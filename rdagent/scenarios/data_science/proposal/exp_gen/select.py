from rdagent.core.proposal import CheckpointSelector, Trace

# # TODO: more advanced selector
# # TODO/Discussion: load selector function here or define selector class in `proposal.py`?


class LatestCKPSelector(CheckpointSelector):
    """
    -`(-1, )` represents starting from the latest trial in the trace
    """

    def get_selection(self, trace: Trace) -> tuple[int, ...]:

        return (-1,)


class SOTAJumpCKPSelector(CheckpointSelector):
    """
    SOTA jump policy:
    if the cumulative SOTA in a window is below a threshold, jump to a new trial
    otherwise, continue the current latest trial
    """

    def __init__(self) -> None:

        self.SOTA_COUNT_WINDOW = 5
        self.SOTA_COUNT_THRESHOLD = 1  # start to compute cumulative SOTA ratio after 5 trials

    def get_selection(self, trace: Trace) -> tuple[int, ...] | None:

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
                return None
            else:
                return (-1,)

        else:
            return (-1,)


# TODO: implement these selectors and more


class GlobalGreedyCKPSelector(CheckpointSelector):
    """
    global greedy selector: select the trial with best performance globally (in trace.hist)
    consistent with the greedy strategy in AIDE
    not implemented yet
    """

    def get_selection(self, trace: Trace) -> tuple[int, ...]:

        return (-1,)


class LocalGreedyCKPSelector(CheckpointSelector):
    """
    local greedy selector: select the trial with best performance locally (in trace.ancestors)
    not implemented yet
    """

    def get_selection(self, trace: Trace) -> tuple[int, ...]:

        return (-1,)


class BugBufferCKPSelector(CheckpointSelector):
    """
    bug buffer selector: with limit-size bug buffer size, start a new trace if buffer exceeds.
    not implemented yet
    """

    def __init__(self) -> None:
        self.bug_count = 0
        self.BUG_BUFFER_SIZE = 10

    def get_selection(self, trace: Trace) -> tuple[int, ...]:

        if self.bug_count < self.BUG_BUFFER_SIZE:
            return (-1,)

        else:
            return None


class RandomCKPSelector(CheckpointSelector):
    def get_selection(self, trace: Trace) -> tuple[int, ...]:
        """
        random selector: select the trial randomly
        not implemented yet
        """
        return (-1,)


class BuggyCKPSelector(CheckpointSelector):
    def get_selection(self, trace: Trace) -> tuple[int, ...]:
        """
        buggy selector: select the most recent trial with buggy performance
        not implemented yet
        """
        return (-1,)
