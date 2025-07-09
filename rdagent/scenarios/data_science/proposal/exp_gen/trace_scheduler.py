from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdagent.scenarios.data_science.proposal.exp_gen.base import DSTrace


class TraceScheduler(ABC):
    """
    An abstract base class for trace scheduling strategies.
    Determines which active trace to expand next during parallel exploration.
    """

    @abstractmethod
    async def next(self, trace: DSTrace) -> tuple[int, ...]:
        """
        Selects the next trace to expand.

        For proposing selections, we have to follow the rules
        - Suggest selection: suggest a selection that is suitable for the current trace.
        - Suggested should be garenteed to be recorded at last!!!
        - If no suitable selection is found, the function should async wait!!!!

        Args:
            trace: The DSTrace object containing the full experiment history.

        Returns:
            A tuple representing the selection of the parent node for the new experiment.
            e.g., (leaf_idx,) for an existing trace, or trace.NEW_ROOT for a new one.
        """
        raise NotImplementedError


class RoundRobinScheduler(TraceScheduler):
    """
    A concurrency-safe scheduling strategy that cycles through active traces
    in a round-robin fashion.

    NOTE: we don't need to use asyncio.Lock here as the kickoff_loop ensures the ExpGen is always sequential, instead of parallel.
    """

    def __init__(self, max_trace_num: int):
        self.max_trace_num = max_trace_num
        self._last_selected_leaf_id = -1
        self.rec_commit_idx = 0  # the node before rec_idx is already committed.
        self.uncommited_rec_status = defaultdict(int)  # the uncommited record status

    async def next(self, trace: DSTrace) -> tuple[int, ...]:
        """
        Atomically selects the next leaf node from the trace in order.
        """
        while True:
            # step 0: Commit the pending selections
            for i in range(self.rec_commit_idx, len(trace.dag_parent)):
                for p in trace.dag_parent[i]:
                    self.uncommited_rec_status[p] -= 1
            self.rec_commit_idx = len(trace.hist)

            # step 1: select the parant trace to expand
            # Policy: if we have fewer traces than our target, start a new one.
            if trace.sub_trace_count + self.uncommited_rec_status[trace.NEW_ROOT] < self.max_trace_num:
                self.uncommited_rec_status[trace.NEW_ROOT] += 1
                return trace.NEW_ROOT

            # Step2: suggest a selection to a not expanding leave
            leaves = trace.get_leaves()
            for leaf in leaves:
                if self.uncommited_rec_status[leaf] == 0:
                    self.uncommited_rec_status[leaf] += 1
                    return (leaf,)
            await asyncio.sleep(1)
