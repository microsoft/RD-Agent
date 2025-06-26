from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdagent.scenarios.data_science.proposal.exp_gen.base import DSTrace


class TraceScheduler(ABC):
    """
    An abstract base class for trace scheduling strategies.
    Determines which active trace to expand next during parallel exploration.
    """

    @abstractmethod
    async def select_trace(self, trace: DSTrace) -> tuple[int, ...]:
        """
        Selects the next trace to expand.

        This method must be async to allow for safe concurrent access.

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

    def __init__(self):
        self._last_selected_leaf_id = -1

    async def select_trace(self, trace: DSTrace) -> tuple[int, ...]:
        """
        Atomically selects the next leaf node from the trace in order.
        """

        leaves = trace.get_leaves()
        if not leaves:
            # This is the very first experiment in a new tree.
            return trace.NEW_ROOT

        # Find the index of the last selected leaf in the current list of leaves
        try:
            current_position = leaves.index(self._last_selected_leaf_id)
            # Move to the next position, wrapping around if necessary
            next_position = (current_position + 1) % len(leaves)
        except ValueError:
            # This can happen if the last selected leaf is no longer a leaf
            # (it has been expanded) or if this is the first selection.
            # In either case, start from the beginning.
            next_position = 0

        selected_leaf = leaves[next_position]
        self._last_selected_leaf_id = selected_leaf

        return (selected_leaf,)
