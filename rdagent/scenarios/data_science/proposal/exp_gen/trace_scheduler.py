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
        self.c_uct = getattr(DS_RD_SETTING, "scheduler_c_uct", 1.0) or 1.0
        # Statistics keyed by leaf node index
        self.node_visit_count: dict[int, int] = {}
        self.node_value_sum: dict[int, float] = {}
        self.node_prior: dict[int, float] = {}

        self.root_id = -1
        self.node_visit_count[self.root_id] = 1
        self.node_value_sum[self.root_id] = 0.0

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

    # def _get_u(self, node_id: int) -> float:
    #     prior = self.node_prior.get(node_id, 0.0)
    #     visits = self.node_visit_count.get(node_id, 0)
    #     # Avoid div-by-zero; encourage exploration when visits are small
    #     return self.c_puct * prior * math.sqrt(max(1, self.global_visit_count)) / (1 + visits)

    def _get_u_uct(self, node_id: int, trace: DSTrace) -> float:
        parents = trace.get_parents(node_id)
        
        #last_parent_id = parents[-2] if len(parents) > 1 else 0
        if len(parents) < 2:
            last_parent_id = self.root_id
        else:
            last_parent_id = parents[-2]

        parent_visits = self.node_visit_count.get(last_parent_id, 0)
        visits = self.node_visit_count.get(node_id, 0)
        N = max(1, parent_visits)
        n = max(1, visits)
        return self.c_uct * math.sqrt(math.log(N) / n)

    def select(self, trace: DSTrace) -> tuple[int, ...] | None:
        # Step 1: keep same policy to reach target number of parallel traces
        # TODO: expanding from the virtual root node is implemented in a rule-based way.

        # if trace.sub_trace_count + self.uncommited_rec_status[trace.NEW_ROOT] < self.max_trace_num:
        #     return trace.NEW_ROOT


        # Step 2: consider only available leaves (not being expanded)
        available_leaves = list(set(range(len(trace.hist))))

        candidates = list(available_leaves)  # copy
        candidates_with_root = candidates + [self.root_id]

        if not available_leaves:
            return None

        # # Step 3: compute priors (P) from potentials via softmax
        # potentials = [self.calculate_potential(trace, leaf) for leaf in available_leaves]
        # if any(p < 0 for p in potentials):
        #     raise ValueError("Potential function returned a negative value.")
        # priors = self._softmax_probabilities(potentials)
        # for leaf, p in zip(available_leaves, priors):
        #     self.node_prior[leaf] = p

        # Step 4: score each leaf using PUCT-like rule: Q + U
        best_leaf = None
        best_score = -float("inf")
        for leaf in candidates_with_root:
            q = self._get_q(leaf)
            #u = self._get_u(leaf)
            #u = self._get_u_uct(leaf,trace)
            u = self._get_u_uct(leaf, trace) if leaf != self.root_id else self.c_uct * math.sqrt(
                    math.log(max(1, self.node_visit_count.get(self.root_id, 1))) / max(1, self.node_visit_count.get(self.root_id, 1))
                )
            score = q + u
            if score > best_score:
                best_score = score
                best_leaf = leaf

        if best_leaf is None:
            return None

        if best_leaf == self.root_id:
            capacity = trace.sub_trace_count + self.uncommited_rec_status.get(trace.NEW_ROOT, 0)
            if capacity >= self.max_trace_num:
                # capacity full: pick next best real leaf
                second_best = None
                second_score = -float("inf")
                for node in candidates:
                    q = self._get_q(node)
                    u = self._get_u_uct(node, trace)
                    score = q + u
                    if score > second_score:
                        second_score = score
                        second_best = node
                if second_best is None:
                    return None
                # optimistic visit update for chosen leaf (optional)
                # self.node_visit_count[second_best] = self.node_visit_count.get(second_best, 0) + 1
                return (second_best,)
            else:
                # choose to expand from virtual root
                # optimistic visit update for root if desired:
                # self.node_visit_count[self.root_id] += 1
                return trace.NEW_ROOT
            
        # # Step 5: optimistic visit update on selection; value update deferred to observe_feedback
        #self.global_visit_count += 1

        return (best_leaf,)
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
        
    def scaled_tanh(self, x):
        # tanh -> (-1,1), then scale to (0,1)
        return (math.tanh(x) + 1.0) / 2.0

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
            if re.result is not None:
                if bigger_is_better:
                    reward = self.scaled_tanh(re.result.loc["ensemble"].iloc[0])
                else:
                    reward = 1- self.scaled_tanh(re.result.loc["ensemble"].iloc[0])
            else:
                reward = 0 if bigger_is_better else 1
        else:
            reward = 1.0 if getattr(fb, "decision", False) else 0.0

        id_list = trace.get_parents(new_idx)
        id_list = [self.root_id] + id_list

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




class MCGSDAGScheduler(ProbabilisticScheduler):
    """
    Monte-Carlo Graph Search Scheduler for multi-parent DAG Trace.
    Uses normal UCT (Q + U) without prior, and fully uses Trace's DAG utilities.
    """

    def __init__(self, max_trace_num: int, c: float = 1.0, *args, **kwargs):
        super().__init__(max_trace_num, temperature=1.0)
        self.c = c
        self.node_visit_count: dict[int, int] = {}
        self.node_value_sum: dict[int, float] = {}
        self.root_id = -1
        self.node_visit_count[self.root_id] = 1
        self.node_value_sum[self.root_id] = 0.0
        self.last_observed_commit_idx = 0

    # -------------------------
    # UCT score
    # -------------------------
    def _get_q(self, node_id: int) -> float:
        visits = self.node_visit_count.get(node_id, 0)
        if visits == 0:
            return 0.0
        return self.node_value_sum.get(node_id, 0.0) / visits

    def _get_u(self, node_id: int, trace) -> float:
        """UCT exploration term using parents in the DAG"""
        parents = trace.dag_parent[node_id] or (self.root_id,)
        parent_visits = sum(self.node_visit_count.get(p, 1) for p in parents)
        n = max(1, self.node_visit_count.get(node_id, 0))
        return self.c * math.sqrt(math.log(parent_visits) / n)

    # -------------------------
    # Node selection
    # -------------------------
    def select(self, trace):
        """
        Select a node to expand based on UCT score.
        Uses all leaves and optionally full path info.
        """
        self.process_uncommitted_nodes(trace)

        leaves = list(range(len(trace.hist)))
        if not leaves:
            return None

        best_score = -1e18
        best_node = None

        for node in leaves + [self.root_id]:
            all_paths = trace.get_all_paths(node) if node != self.root_id else [[self.root_id]]
            q_values = [sum(self._get_q(n) for n in path) / len(path) for path in all_paths]
            avg_q = sum(q_values) / len(q_values)
            u = self._get_u(node, trace)
            score = avg_q + u
            if score > best_score:
                best_score = score
                best_node = node

        # Handle NEW_ROOT
        if best_node == self.root_id:
            capacity = trace.sub_trace_count + self.uncommited_rec_status.get(trace.NEW_ROOT, 0)
            if capacity >= self.max_trace_num:
                # pick best leaf instead
                best_leaf = max(leaves, key=lambda n: self._get_q(n) + self._get_u(n, trace))
                return (best_leaf,)
            else:
                return trace.NEW_ROOT

        return (best_node,)

    # -------------------------
    # Backpropagation
    # -------------------------
    def observe_feedback(self, trace, idx):
        """
        Update Q/N values along all ancestors using
        """
        re, fb = trace.hist[idx]
        reward = 1.0 if getattr(fb, "decision", False) else 0.0
        ancestors = trace.get_parents_dag(idx) + [idx]  # ancestors first, then self
        for node in ancestors:
            self.node_visit_count[node] = self.node_visit_count.get(node, 0) + 1
            self.node_value_sum[node] = self.node_value_sum.get(node, 0.0) + reward

    # -------------------------
    # Batch feedback
    # -------------------------
    def process_uncommitted_nodes(self, trace):
        start = self.last_observed_commit_idx
        end = min(len(trace.dag_parent), len(trace.hist))
        for i in range(start, end):
            self.observe_feedback(trace, i)
        self.last_observed_commit_idx = end

    # -------------------------
    # Reset
    # -------------------------
    def reset(self):
        super().reset()
        self.node_visit_count.clear()
        self.node_value_sum.clear()
        self.last_observed_commit_idx = 0
        self.node_visit_count[self.root_id] = 1
        self.node_value_sum[self.root_id] = 0.0
