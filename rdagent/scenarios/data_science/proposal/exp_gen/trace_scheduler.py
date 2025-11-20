from __future__ import annotations

import asyncio
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.kaggle.kaggle_crawler import get_metric_direction

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

    def reset(self) -> None:
        """
        Reset the scheduler to the initial state.
        """
        pass


class BaseScheduler(TraceScheduler):
    def __init__(self):
        self.rec_commit_idx = 0  # the node before rec_idx is already committed.
        self.uncommited_rec_status = defaultdict(int)  # the uncommited record status

    async def next(self, trace: DSTrace) -> tuple[int, ...]:
        """
        Atomically selects the next leaf node from the trace in order.
        """
        while True:
            # step 1: Commit the pending selections
            self.process_uncommitted_nodes(trace)

            # step 2: update uncommited_rec_status & rec_commit_idx
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

    def process_uncommitted_nodes(self, trace: DSTrace) -> None:
        """
        A slot for implementing custom logic to process uncommitted nodes.

        `uncommited_rec_status` & `rec_commit_idx` will be updated automatically.
        """

    @abstractmethod
    def select(self, trace: DSTrace) -> tuple[int, ...] | None:
        """Selects the parent nodes for the new experiment, or None if no selection can be made."""
        raise NotImplementedError

    def reset(self) -> None:
        self.uncommited_rec_status = defaultdict(int)
        self.rec_commit_idx = 0


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


class MCTSScheduler(ProbabilisticScheduler):
    """
    A simplified MCTS-based scheduler using a PUCT-like scoring rule.

    Formula:
    U(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
    where Q is the average reward, N is the visit count, P is the prior probability, c_puct is the given weight to balance exploration and exploitation.

    Design goals for the initial version:
    - Reuse ProbabilisticScheduler's potential calculation as prior P (via softmax).
    - Maintain visit/value statistics per leaf to compute Q and U.
    - Update visits on selection; update values after feedback via observe_feedback.
    - Keep NEW_ROOT policy and uncommitted status handling identical to base classes.
    """

    def __init__(self, max_trace_num: int, temperature: float = 1.0, *args, **kwargs):
        super().__init__(max_trace_num, temperature)
        # Read c_puct from settings if available, otherwise fall back to default 1.0
        self.c_puct = getattr(DS_RD_SETTING, "scheduler_c_puct", 1.0) or 1.0
        # Statistics keyed by leaf node index
        self.node_visit_count: dict[int, int] = {}
        self.node_value_sum: dict[int, float] = {}
        self.node_prior: dict[int, float] = {}
        # Global counter to stabilize U term
        self.global_visit_count: int = 0
        # Last observed commit index for batch feedback observation
        self.last_observed_commit_idx: int = 0

    def _get_q(self, node_id: int) -> float:
        visits = self.node_visit_count.get(node_id, 0)
        value_sum = self.node_value_sum.get(node_id, 0.0)
        if visits <= 0:
            # Unseen nodes default to neutral Q
            return 0.0
        return value_sum / visits

    def _get_u(self, node_id: int) -> float:
        prior = self.node_prior.get(node_id, 0.0)
        visits = self.node_visit_count.get(node_id, 0)
        # Avoid div-by-zero; encourage exploration when visits are small
        return self.c_puct * prior * math.sqrt(max(1, self.global_visit_count)) / (1 + visits)

    def select(self, trace: DSTrace) -> tuple[int, ...] | None:
        # Step 1: keep same policy to reach target number of parallel traces
        # TODO: expanding from the virtual root node is implemented in a rule-based way.
        if trace.sub_trace_count + self.uncommited_rec_status[trace.NEW_ROOT] < self.max_trace_num:
            return trace.NEW_ROOT

        # Step 2: consider only available leaves (not being expanded)
        available_leaves = list(set(range(len(trace.hist))))
        if not available_leaves:
            return None

        # Step 3: compute priors (P) from potentials via softmax
        potentials = [self.calculate_potential(trace, leaf) for leaf in available_leaves]
        if any(p < 0 for p in potentials):
            raise ValueError("Potential function returned a negative value.")
        priors = self._softmax_probabilities(potentials)
        for leaf, p in zip(available_leaves, priors):
            self.node_prior[leaf] = p

        # Step 4: score each leaf using PUCT-like rule: Q + U
        best_leaf = None
        best_score = -float("inf")
        for leaf in available_leaves:
            q = self._get_q(leaf)
            u = self._get_u(leaf)
            score = q + u
            if score > best_score:
                best_score = score
                best_leaf = leaf

        if best_leaf is None:
            return None

        # # Step 5: optimistic visit update on selection; value update deferred to observe_feedback
        self.global_visit_count += 1

        return (best_leaf,)

    def observe_feedback(self, trace: DSTrace, new_idx: int) -> None:
        """
        Update statistics after an experiment is committed to the trace.

        Args:
            trace: The DSTrace object.
            new_idx: Index of the newly appended experiment in trace.hist.
            reward: Optional explicit reward. If None, derive from feedback.decision (1.0/0.0).
        """

        re, fb = trace.hist[new_idx]
        if DS_RD_SETTING.enable_score_reward:
            bigger_is_better = get_metric_direction(trace.scen.competition)
            if getattr(fb, "decision", False):
                reward = math.tanh(re.result.loc["ensemble"].iloc[0].round(3)) * (1 if bigger_is_better else -1)
            else:
                reward = -1 if bigger_is_better else 1
        else:
            reward = 1.0 if getattr(fb, "decision", False) else 0.0
        id_list = trace.get_parents(new_idx)
        for id in id_list:
            self.node_value_sum[id] = self.node_value_sum.get(id, 0.0) + float(reward)
            self.node_visit_count[id] = self.node_visit_count.get(id, 0) + 1

    def reset(self) -> None:
        """
        Clear all maintained statistics. Should be called when the underlying trace is reset.
        """
        super().reset()
        self.node_visit_count.clear()
        self.node_value_sum.clear()
        self.node_prior.clear()
        self.global_visit_count = 0
        self.last_observed_commit_idx = 0

    def process_uncommitted_nodes(self, trace: DSTrace) -> None:
        """
        Batch observe all newly committed experiments since last observation.
        Should be called before making a new selection to ensure statistics are up-to-date.
        """
        start_idx = max(0, self.last_observed_commit_idx)
        # Only observe fully committed items (both dag_parent and hist appended)
        end_idx = min(len(trace.dag_parent), len(trace.hist))
        if start_idx >= end_idx:
            return
        for idx in range(start_idx, end_idx):
            self.observe_feedback(trace, idx)
        self.last_observed_commit_idx = end_idx
