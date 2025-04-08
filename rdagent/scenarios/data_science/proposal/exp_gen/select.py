from rdagent.core.proposal import CheckpointSelector, Trace

# # TODO: more advanced selector
# # TODO/Discussion: load selector function here or define selector class in `proposal.py`?


class LatestCKPSelector(CheckpointSelector):
    """
    -`(-1, )` represents starting from the latest trial in the trace
    """

    def get_selection(self, trace: Trace) -> tuple[int, ...]:

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
