from __future__ import annotations

import asyncio
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

from rdagent.log import rdagent_logger as logger

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
        - Suggested should be garenteed to be recorded at last!!!!
        - If no suitable selection is found, the function should async wait!!!!

        Args:
            trace: The DSTrace object containing the full experiment history.

        Returns:
            A tuple representing the selection of the parent node for the new experiment.
            e.g., (leaf_idx,) for an existing trace, or trace.NEW_ROOT for a new one.
        """
        raise NotImplementedError


class BaseScheduler(TraceScheduler):
    def __init__(self):
        self.rec_commit_idx = 0  # the node before rec_idx is already committed.
        self.uncommited_rec_status = defaultdict(int)  # the uncommited record status

    async def next(self, trace: DSTrace) -> tuple[int, ...]:
        """
        Atomically selects the next leaf node from the trace in order.
        """
        while True:
            # step 0: Commit the pending selections
            for i in range(self.rec_commit_idx, len(trace.dag_parent)):
                parent_of_i = trace.dag_parent[i]
                if parent_of_i == trace.NEW_ROOT:
                    self.uncommited_rec_status[trace.NEW_ROOT] -= 1
                else:
                    for p in parent_of_i:
                        self.uncommited_rec_status[p] -= 1
            self.rec_commit_idx = len(trace.hist)

            parents = self.select(trace)

            if parents is not None:
                if parents == trace.NEW_ROOT:
                    self.uncommited_rec_status[trace.NEW_ROOT] += 1
                else:
                    for p in parents:
                        self.uncommited_rec_status[p] += 1
                return parents

            await asyncio.sleep(1)

    @abstractmethod
    def select(self, trace: DSTrace) -> tuple[int, ...] | None:
        """Selects the parent nodes for the new experiment, or None if no selection can be made."""
        raise NotImplementedError


class RoundRobinScheduler(BaseScheduler):
    """
    A concurrency-safe scheduling strategy that cycles through active traces
    in a round-robin fashion.

    NOTE: we don't need to use asyncio.Lock here as the kickoff_loop ensures the ExpGen is always sequential, instead of parallel.
    """

    def __init__(self, max_trace_num: int, *args, **kwargs):
        logger.info(f"RoundRobinScheduler: max_trace_num={max_trace_num}")
        self.max_trace_num = max_trace_num
        self._last_selected_leaf_id = -1
        super().__init__()

    def select(self, trace: DSTrace) -> tuple[int, ...] | None:
        """
        Atomically selects the next leaf node from the trace in order.
        If no suitable selection is found, return None.
        """
        # Policy: if we have fewer traces than our target, start a new one.
        if trace.sub_trace_count + self.uncommited_rec_status[trace.NEW_ROOT] < self.max_trace_num:
            return trace.NEW_ROOT

        # Step2: suggest a selection to a not expanding leave
        leaves = trace.get_leaves()
        for leaf in leaves:
            if self.uncommited_rec_status[leaf] == 0:
                return (leaf,)

        return None


# ======================================================================================
# Probabilistic Scheduler and its potential functions
# ======================================================================================


class ProbabilisticScheduler(BaseScheduler):
    """
    A concurrency-safe scheduling strategy that samples the next trace to expand
    based on a probability distribution derived from a potential function.
    """

    def __init__(self, max_trace_num: int, temperature: float = 1.0, *args, **kwargs):
        """
        Args:
            max_trace_num: The target number of parallel traces.
            temperature: Temperature parameter for softmax calculation. Higher values make selection more uniform.
        """
        if max_trace_num <= 0:
            raise ValueError("max_trace_num must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.max_trace_num = max_trace_num
        self.temperature = temperature
        super().__init__()

    def calculate_potential(self, trace: DSTrace, leaf_id: int) -> float:
        """
        Calculate potential score for a given leaf node.
        This is the base implementation that provides uniform distribution.

        Args:
            trace: The DSTrace object containing the full experiment history.
            leaf_id: The index of the leaf node to evaluate.

        Returns:
            float: A potential score. Higher means more likely to be selected.
        """
        return 1.0  # Uniform distribution by default

    def _softmax_probabilities(self, potentials: list[float]) -> list[float]:
        """
        Convert potential scores to probabilities using softmax.

        Args:
            potentials: List of potential scores.

        Returns:
            List of probabilities that sum to 1.
        """
        if not potentials:
            return []

        # Apply temperature scaling
        scaled_potentials = [p / self.temperature for p in potentials]

        # Compute softmax
        max_potential = max(scaled_potentials)
        exp_potentials = [math.exp(p - max_potential) for p in scaled_potentials]
        sum_exp = sum(exp_potentials)

        if sum_exp == 0:
            # If all potentials are very small, return uniform distribution
            return [1.0 / len(potentials)] * len(potentials)

        return [exp_p / sum_exp for exp_p in exp_potentials]

    def select(self, trace: DSTrace) -> tuple[int, ...] | None:
        """
        Selects the next leaf node based on probabilistic sampling.
        """
        # Step 1: If we have fewer traces than our target, start a new one.
        # This policy prioritizes reaching the desired number of traces.
        if trace.sub_trace_count + self.uncommited_rec_status[trace.NEW_ROOT] < self.max_trace_num:
            return trace.NEW_ROOT

        # Step 2: Probabilistically select a leaf to expand.
        leaves = trace.get_leaves()
        available_leaves = [leaf for leaf in leaves if self.uncommited_rec_status[leaf] == 0]

        if not available_leaves:
            return None

        # Calculate potential for each available leaf
        potentials = [self.calculate_potential(trace, leaf) for leaf in available_leaves]

        if any(p < 0 for p in potentials):
            raise ValueError("Potential function returned a negative value.")

        # Convert potentials to probabilities using softmax
        probabilities = self._softmax_probabilities(potentials)

        # Select a leaf based on probabilities
        selected_leaf = random.choices(available_leaves, weights=probabilities, k=1)[0]

        return (selected_leaf,)


class TraceLengthScheduler(ProbabilisticScheduler):
    """
    A scheduler that prefers longer traces (more experiments)
      -- default: prefer to expand the trace that has more experiments (quicker to get the result).
      -- if inverse=True, prefer to expand the trace that has less experiments.

    """

    def __init__(self, max_trace_num: int, temperature: float = 1.0, inverse: bool = False, *args, **kwargs):
        """
        Args:
            max_trace_num: The target number of parallel traces.
            temperature: Temperature parameter for softmax calculation.
            inverse: If True, shorter traces get higher potential.
        """
        logger.info(
            f"TraceLengthScheduler: max_trace_num={max_trace_num}, temperature={temperature}, inverse={inverse}"
        )
        super().__init__(max_trace_num, temperature)
        self.inverse = inverse

    def calculate_potential(self, trace: DSTrace, leaf_id: int) -> float:
        """
        Calculate potential based on the length of the trace leading to the leaf.
        """
        # Get the path from root to this leaf using existing method
        path = trace.get_parents(leaf_id)
        path_len = len(path)

        if path_len == 0:
            return 1.0

        return 1.0 / path_len if self.inverse else float(path_len)


class SOTABasedScheduler(ProbabilisticScheduler):
    """
    A scheduler that prefers traces with more SOTA (State of the Art) results.
    """

    def __init__(self, max_trace_num: int, temperature: float = 1.0, inverse: bool = False, *args, **kwargs):
        """
        Args:
            max_trace_num: The target number of parallel traces.
            temperature: Temperature parameter for softmax calculation.
            inverse: If True, fewer SOTA results get higher potential.
        """
        logger.info(f"SOTABasedScheduler: max_trace_num={max_trace_num}, temperature={temperature}, inverse={inverse}")
        super().__init__(max_trace_num, temperature)
        self.inverse = inverse

    def calculate_potential(self, trace: DSTrace, leaf_id: int) -> float:
        """
        Calculate potential based on the number of SOTA results in the trace.
        """
        # Get the path from root to this leaf
        path = trace.get_parents(leaf_id)
        sota_count = 0

        for node_id in path:
            # Check if this experiment was successful (decision=True)
            if node_id < len(trace.hist):
                exp, feedback = trace.hist[node_id]
                if feedback.decision:
                    sota_count += 1

        if self.inverse:
            # Add 1 to avoid division by zero and give traces with 0 SOTAs the highest potential.
            return 1.0 / (sota_count + 1)
        return float(sota_count)


class RandomScheduler(ProbabilisticScheduler):
    """
    A scheduler that selects traces randomly with uniform distribution.
    """

    def calculate_potential(self, trace: DSTrace, leaf_id: int) -> float:
        """
        Return random potential for uniform random selection.
        """
        return random.random()
