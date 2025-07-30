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

                if trace.dag_parent[i] == trace.NEW_ROOT:
                    self.uncommited_rec_status[trace.NEW_ROOT] -= 1
                else:
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


# ======================================================================================
# Probabilistic Scheduler and its potential functions
# ======================================================================================

import random
from typing import Callable


if TYPE_CHECKING:
    from rdagent.scenarios.data_science.proposal.exp_gen.base import DSTrace


PotentialFunc = Callable[[DSTrace, int], float]
"""
A function that calculates a potential score for a given trace leaf.
Args:
    trace (DSTrace): The entire experiment history.
    leaf_id (int): The index of the leaf node to evaluate.
Returns:
    float: A non-negative potential score. Higher means more likely to be selected.
"""


def random_potential(trace: DSTrace, leaf_id: int) -> float:
    """Assigns a random potential, leading to uniform random selection."""
    return random.random()


def trace_length_potential(trace: DSTrace, leaf_id: int, inverse: bool = False) -> float:
    """
    Calculates potential based on the length of the trace leading to the leaf.
    Longer traces get higher potential by default.
    If inverse=True, shorter traces get higher potential.
    NOTE: This is an example and assumes `trace.get_path_to_root(leaf_id)` exists.
    """
    # This is a hypothetical way to get the path length.
    path_len = len(trace.get_path_to_root(leaf_id))
    if path_len == 0:
        return 1.0
    return 1.0 / path_len if inverse else float(path_len)


def sota_potential(trace: DSTrace, leaf_id: int) -> float:
    """
    Calculates potential based on the number of SOTA results in the trace.
    NOTE: This is an example and assumes `trace.hist[i]['is_sota']` exists
    and `trace.get_path_to_root(leaf_id)` exists.
    The actual implementation will depend on the structure of `DSTrace`.
    """
    path = trace.get_path_to_root(leaf_id)
    sota_count = 0
    for node_id in path:
        # This is a hypothetical way to check for SOTA.
        if trace.hist[node_id].get("is_sota", False):
            sota_count += 1
    return float(sota_count)


class ProbabilisticScheduler(TraceScheduler):
    """
    A concurrency-safe scheduling strategy that samples the next trace to expand
    based on a probability distribution derived from a potential function.
    """

    def __init__(self, max_trace_num: int, potential_fn: PotentialFunc = random_potential):
        """
        Args:
            max_trace_num: The target number of parallel traces.
            potential_fn: A function that takes (DSTrace, leaf_id) and returns a
                          float potential score. Defaults to random potential.
        """
        if max_trace_num <= 0:
            raise ValueError("max_trace_num must be positive.")
        self.max_trace_num = max_trace_num
        self.potential_fn = potential_fn
        self.rec_commit_idx = 0
        self.uncommited_rec_status = defaultdict(int)

    async def next(self, trace: DSTrace) -> tuple[int, ...]:
        """
        Selects the next leaf node based on probabilistic sampling.
        """
        while True:
            # Step 0: Commit the pending selections. This logic is identical to
            # RoundRobinScheduler as it's about state synchronization.
            for i in range(self.rec_commit_idx, len(trace.dag_parent)):
                if trace.dag_parent[i] == trace.NEW_ROOT:
                    self.uncommited_rec_status[trace.NEW_ROOT] -= 1
                else:
                    for p in trace.dag_parent[i]:
                        self.uncommited_rec_status[p] -= 1
            self.rec_commit_idx = len(trace.hist)

            # Step 1: If we have fewer traces than our target, start a new one.
            # This policy prioritizes reaching the desired number of traces.
            if trace.sub_trace_count + self.uncommited_rec_status[trace.NEW_ROOT] < self.max_trace_num:
                self.uncommited_rec_status[trace.NEW_ROOT] += 1
                return trace.NEW_ROOT

            # Step 2: Probabilistically select a leaf to expand.
            leaves = trace.get_leaves()
            available_leaves = [leaf for leaf in leaves if self.uncommited_rec_status[leaf] == 0]

            if not available_leaves:
                await asyncio.sleep(1)
                continue

            # Calculate potential for each available leaf
            potentials = [self.potential_fn(trace, leaf) for leaf in available_leaves]

            if any(p < 0 for p in potentials):
                raise ValueError("Potential function returned a negative value.")

            total_potential = sum(potentials)

            # Select a leaf. If total potential is 0, select uniformly.
            if total_potential > 0:
                selected_leaf = random.choices(available_leaves, weights=potentials, k=1)[0]
            else:
                selected_leaf = random.choice(available_leaves)

            # Mark the selected leaf as "pending expansion"
            self.uncommited_rec_status[selected_leaf] += 1
            return (selected_leaf,)
