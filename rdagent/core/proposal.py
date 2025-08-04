# TODO: remove `self.scen` if traces will be passed into the instance.

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evaluation import Feedback
from rdagent.core.experiment import ASpecificExp, Experiment
from rdagent.core.knowledge_base import KnowledgeBase
from rdagent.core.scenario import Scenario

import torch
from typing import Generic, Optional,Callable
import itertools
from rdagent.core.mcts_config import SearchAlgorithm,MCTSConfig
from typing import Generic, Optional, NamedTuple, Callable, Union
import math
import numpy as np
from copy import deepcopy

if TYPE_CHECKING:
    from rdagent.utils.workflow.loop import LoopBase


class Hypothesis:
    """
    TODO: We may have better name for it.

    Name Candidates:
    - Belief
    """

    def __init__(
        self,
        hypothesis: str,
        reason: str,
        concise_reason: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
    ) -> None:
        self.hypothesis: str = hypothesis
        self.reason: str = reason
        self.concise_reason: str = concise_reason
        self.concise_observation: str = concise_observation
        self.concise_justification: str = concise_justification
        self.concise_knowledge: str = concise_knowledge

    def __str__(self) -> str:
        return f"""Hypothesis: {self.hypothesis}
Reason: {self.reason}"""

    # source: data_ana | model_nan = None


# Origin(path of repo/data/feedback) => view/summarization => generated Hypothesis


class ExperimentFeedback(Feedback):
    def __init__(
        self,
        reason: str,
        *,
        code_change_summary: str | None = None,
        decision: bool,
        eda_improvement: str | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.decision = decision
        self.eda_improvement = eda_improvement
        self.reason = reason
        # Exception is not None means failing to generate runnable experiments due to exception.
        # Runable reuslts are not always good.
        self.exception: Exception | None = (
            exception  # if the experiment raises exception, it will be integrated into part of the feedback.
        )
        self.code_change_summary = code_change_summary

    def __bool__(self) -> bool:
        return self.decision

    def __str__(self) -> str:
        res = f"Decision: {self.decision}\nReason: {self.reason}"
        code_change_summary = getattr(self, "code_change_summary", None)
        if code_change_summary is not None:
            res += "\nCode Change Summary: " + code_change_summary
        return res

    @classmethod
    def from_exception(cls, e: Exception) -> ExperimentFeedback:
        """
        A convenient method to create Feedback from an exception.
        """
        return cls(decision=False, reason=f"The experiment fails due to {e!s}", exception=e)


class HypothesisFeedback(ExperimentFeedback):
    def __init__(
        self,
        observations: str,
        hypothesis_evaluation: str,
        new_hypothesis: str,
        reason: str,
        *,
        code_change_summary: str | None = None,
        decision: bool,
        eda_improvement: str | None = None,
    ) -> None:
        super().__init__(
            reason,
            decision=decision,
            code_change_summary=code_change_summary,
            eda_improvement=eda_improvement,
        )
        self.observations = observations
        self.hypothesis_evaluation = hypothesis_evaluation
        self.new_hypothesis = new_hypothesis

    def __str__(self) -> str:
        return f"""{super().__str__()}
Observations: {self.observations}
Hypothesis Evaluation: {self.hypothesis_evaluation}
New Hypothesis: {self.new_hypothesis}"""


ASpecificScen = TypeVar("ASpecificScen", bound=Scenario)
ASpecificKB = TypeVar("ASpecificKB", bound=KnowledgeBase)

State = TypeVar("State") # loop id
Action = TypeVar("Action") # get hypo/problem list 

class Trace(Generic[ASpecificScen, ASpecificKB]):
    NodeType = tuple[Experiment, ExperimentFeedback]  # Define NodeType as a new type representing the tuple
    NEW_ROOT: tuple = ()

    def __init__(self, scen: ASpecificScen, knowledge_base: ASpecificKB | None = None) -> None:
        self.scen: ASpecificScen = scen
        self.hist: list[Trace.NodeType] = (
            []
        )  # List of tuples containing experiments and their feedback, organized over time.
        self.dag_parent: list[tuple[int, ...]] = []  # List of tuples representing parent indices in the DAG structure.
        # (,) represents no parent; (1,) presents one parent; (1, 2) represents two parents.

        # TODO: self.hist is 2-tuple now, remove hypothesis from it, change old code for this later.
        self.knowledge_base: ASpecificKB | None = knowledge_base
        self.current_selection: tuple[int, ...] = (-1,)

    def get_sota_hypothesis_and_experiment(self) -> tuple[Hypothesis | None, Experiment | None]:
        """Access the last experiment result, sub-task, and the corresponding hypothesis."""
        # TODO: The return value does not align with the signature.
        for experiment, feedback in self.hist[::-1]:
            if feedback.decision:
                return experiment.hypothesis, experiment

        return None, None

    def is_selection_new_tree(self, selection: tuple[int, ...] | None = None) -> bool:
        """
        Check if the current trace is a new tree.
        - selection maybe (-1,) when the dag_parent is empty.
        """
        if selection is None:
            selection = self.get_current_selection()

        return selection == self.NEW_ROOT or len(self.dag_parent) == 0

    def get_current_selection(self) -> tuple[int, ...]:
        return self.current_selection

    def set_current_selection(self, selection: tuple[int, ...]) -> None:
        self.current_selection = selection

    def get_parent_exps(
        self,
        selection: tuple[int, ...] | None = None,
    ) -> list[Trace.NodeType]:
        """
        Collect all ancestors of the given selection.
        The return list follows the order of [root->...->parent->current_node].
        """
        if selection is None:
            selection = self.get_current_selection()

        if self.is_selection_new_tree(selection):
            return []

        return [self.hist[i] for i in self.get_parents(selection[0])]

    def exp2idx(self, exp: Experiment | list[Experiment]) -> int | list[int] | None:
        if isinstance(exp, list):
            exps: list[Experiment] = exp

            # keep the order
            exp_to_index: dict[Experiment, int] = {_exp: i for i, (_exp, _) in enumerate(self.hist)}
            return [exp_to_index[_exp] for _exp in exps]
        for i, (_exp, _) in enumerate(self.hist):
            if _exp == exp:
                return i
        return None

    def idx2exp(self, idx: int | list[int]) -> Experiment | list[Experiment]:
        if isinstance(idx, list):
            idxs: list[int] = idx
            return [self.hist[_idx][0] for _idx in idxs]
        return self.hist[idx][0]

    def is_parent(self, parent_idx: int, child_idx: int) -> bool:
        ancestors = self.get_parents(child_idx)
        return parent_idx in ancestors

    def get_parents(self, child_idx: int) -> list[int]:
        if self.is_selection_new_tree((child_idx,)):
            return []

        ancestors: list[int] = []
        curr = child_idx
        while True:
            ancestors.insert(0, curr)
            parent_tuple = self.dag_parent[curr]
            if not parent_tuple or parent_tuple[0] == curr:
                break
            curr = parent_tuple[0]

        return ancestors


class CheckpointSelector:
    """
    In the trace, we may start from any check point (we'll represent it as a variable `from_checkpoint_idx`)
    """

    @abstractmethod
    def get_selection(self, trace: Trace) -> tuple[int, ...] | None:
        """
        checkpoint_idx represents the place where we want to create a new node.
        the return value should be the idx of target node (the parent of the new generating node).
        - `(-1, )` represents starting from the latest trial in the trace - default value
        - `(idx, )` represents starting from the `idx`-th trial in the trace.
        - `None` represents starting from scratch (start a new trace)

        - More advanced selection strategies in `select.py`
        """


class SOTAexpSelector:
    """
    Select the SOTA experiment from the trace to submit
    """

    @abstractmethod
    def get_sota_exp_to_submit(self, trace: Trace) -> Experiment | None:
        """
        Select the SOTA experiment from the trace to submit
        """


class ExpGen(ABC):

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def gen(self, trace: Trace) -> Experiment:
        """
        Generate the experiment based on the trace.

        `ExpGen().gen()` play a role like

        .. code-block:: python

            # ExpGen().gen() ==
            Hypothesis2Experiment().convert(
                HypothesisGen().gen(trace)
            )
        """

    async def async_gen(self, trace: Trace, loop: LoopBase) -> Experiment:
        """
        generate the experiment and decide whether to stop yield generation and give up control to other routines.
        """
        # we give a default implementation here.
        # The proposal is set to try best to generate the experiment in max-parallel level.
        while True:
            if loop.get_unfinished_loop_cnt(loop.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                return self.gen(trace)
            await asyncio.sleep(1)


class HypothesisGen(ABC):

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def gen(self, trace: Trace) -> Hypothesis:
        # def gen(self, scenario_desc: str, ) -> Hypothesis:
        """
        Motivation of the variable `scenario_desc`:
            - Mocking a data-scientist is observing the scenario.

        scenario_desc may include:
            - data observation:
                - Original or derivative
            - Task information:
        """


class Hypothesis2Experiment(ABC, Generic[ASpecificExp]):
    """
    [Abstract description => concrete description] => Code implementation Card
    """

    @abstractmethod
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> ASpecificExp:
        """Connect the idea proposal to implementation"""
        ...


# Boolean, Reason, Confidence, etc.


class Experiment2Feedback(ABC):
    """ "Generated feedbacks on the hypothesis from **Executed** Implementations of different tasks
    & their comparisons with previous performances"""

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def generate_feedback(self, exp: Experiment, trace: Trace) -> ExperimentFeedback:
        """
        The `exp` should be executed and the results should be included, as well as the comparison
        between previous results (done by LLM).
        For example: `mlflow` of Qlib will be included.
        """
        error_message = "generate_feedback method is not implemented."
        raise NotImplementedError(error_message)


# MCTS is from https://github.com/YuxiXie/MCTS-DPO


class MCTSNode(Generic[State, Action]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self, 
        state: Optional[State], 
        action: Optional[Action], 
        parent: "Optional[MCTSNode]" = None,
        base_rewards: torch.Tensor = None, 
        value: float = 0.0, 
        embeddings: torch.Tensor = None, 
        log_probs: torch.Tensor = None,
        ref_log_probs: torch.Tensor = None,
        is_terminal: bool = False,
        length_penalty: float = 1.25,
    ):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param embeddings: the embeddings of the current state (BERTScore calculation for similar generations filtering)
        :param is_terminal: whether the current state is a terminal state
        
        :param rewards: base rewards
        :param value: advantage of taking the action
        """
        self.exp = None
        self.feedback = None  # type: ExperimentFeedback | None
        self.id = next(MCTSNode.id_iter)
        self.is_terminal = is_terminal
        self.state = state
        self.action = action
        self.parent = parent
        self.embeddings = embeddings
        self.children: 'Optional[list[MCTSNode]]' = None
        self.depth = 0 if parent is None else parent.depth + 1
        self.length_penalty = length_penalty
        
        self.rewards = base_rewards
        self.log_probs = log_probs
        self.ref_log_probs = ref_log_probs
        self.value = value
        
        self.N = 0
        self.V = 0.0
        self.Q = self.parent.V + self.r if self.parent is not None else self.r

    @property
    def r(self) -> float:
        if self.rewards is None:
            return self.value if self.parent is None else (self.value - self.parent.value)
        
        # 默认处理：直接用 rewards.mean()
        return self.rewards.mean().item()
        
    @property
    def p(self) -> float:
        return 1#(self.log_probs.sum() / self.log_probs.size(-1) ** self.length_penalty).exp().detach().item()




class MCTSTrace(Generic[ASpecificScen, ASpecificKB]):
    NodeType = tuple[Experiment, ExperimentFeedback, MCTSNode]  # Define NodeType as a new type representing the tuple
    NEW_ROOT: tuple = ()


    def __init__(self, scen: ASpecificScen, args: MCTSConfig, knowledge_base: ASpecificKB | None = None
    ) -> None:
        self.scen: ASpecificScen = scen
        self.hist: list[Trace.NodeType] = (
            []
        )  # List of tuples containing experiments and their feedback, organized over time.
        self.dag_parent: list[tuple[int, ...]] = []  # List of tuples representing parent indices in the DAG structure.
        # (,) represents no parent; (1,) presents one parent; (1, 2) represents two parents.

        # TODO: self.hist is 2-tuple now, remove hypothesis from it, change old code for this later.
        self.knowledge_base: ASpecificKB | None = knowledge_base
        self.current_selection: tuple[int, ...] = (-1,)

        self.world_model = None
        self.search_config = None
        self.output_trace_in_each_iter = args.output_trace_in_each_iter
        self.w_exp = args.w_exp
        self.depth_limit = args.depth_limit
        self.breadth_limit = args.breadth_limit
        self.n_iters = args.n_iters
        self.gamma = args.gamma
        self.add_kl = args.add_kl
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(args.simulate_strategy,
                                                                                             args.simulate_strategy)
        self.temperature = args.temperature
        self.temperature_decay_ratio = args.temperature_decay_ratio
        self.follow_probability = False
        self._output_iter: list[MCTSNode] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.root: Optional[MCTSNode] = None
        self.disable_tqdm = args.disable_tqdm
        self.consider_diversity = args.consider_diversity
        self.length_penalty = args.length_penalty

        self.policy_model = None

    def calculate_diversity_score(self,candidates):
        if candidates is None: return 0
        
        Q_values = [sample.Q for sample in candidates]
        variance = np.var(np.asarray(Q_values))
        gap = max(Q_values) - min(Q_values)
        # return gap if max(Q_values) > 0 else gap * 0.5
        
        visit_counts = [sample.N for sample in candidates]
        gap = max(visit_counts) - min(visit_counts)
        return gap

    def _get_simulated_pi(self, cur_node: MCTSNode, return_selection=False) -> list[float]:
        """
        Apated from: https://github.com/suragnair/alpha-zero-general/blob/ce020c8eebbabf0e22654279508a6887b4791015/MCTS.py#L28C5-L53C21
        """
        visit_counts = [child.N for child in cur_node.children]
        next_action_V = [child.V for child in cur_node.children]
        next_action_Q = [child.Q for child in cur_node.children]
        next_action_n_children = [len(child.children) if child.children is not None else 0 for child in cur_node.children]
        next_action_variance = [self.calculate_diversity_score(child.children) for child in cur_node.children]
        
        def _cal_probs(temp):
            if temp > 0:
                try:
                    ## choice 1: to sample based on visit counts
                    # counts = [(x * (nc + 1 if self.consider_diversity else 1)) ** (1. / temp) if x else x \
                    #     for x, nc in zip(visit_counts, next_action_n_children)]
                    ## choice 2: to sample based on Q values
                    counts = [(math.exp(x) * (nc + 1 if self.consider_diversity else 1)) ** (1. / temp) if x else x \
                        for x, nc in zip(next_action_Q, next_action_n_children)]
                    total_count = float(sum(counts))
                    probs = [x / total_count for x in counts]
                    return probs
                except OverflowError as e:
                    print(('Run into {} -- Temperature too small ... Set to zero ...').format(str(e)))
            best_actions = np.array(np.argwhere(visit_counts == np.max(visit_counts))).flatten()
            probs = [0] * len(visit_counts)
            for best_action in best_actions:
                probs[best_action] = 1 / len(best_actions)
            return probs
        
        temperature = self.temperature * (self.temperature_decay_ratio ** cur_node.depth)
        probs = _cal_probs(temperature)
        
        if return_selection:
            if temperature == 0:
                ## choice 1: to sample based on visit counts
                # selected_idx = max(range(len(visit_counts)), key=lambda x: (
                #     (next_action_Q[x] + 2) * (next_action_variance[x] + 1 if self.consider_diversity else 1), 
                #     visit_counts[x], next_action_V[x]
                # ))
                ## choice 2: to sample based on Q values
                selected_idx = max(range(len(visit_counts)), key=lambda x: (
                    visit_counts[x] * (next_action_variance[x] + 1 if self.consider_diversity else 1), 
                    next_action_Q[x], next_action_V[x]
                ))
            else:
                selected_idx = np.random.choice(range(len(visit_counts)), p=probs)
            return probs, selected_idx, next_action_V, next_action_Q
        return probs, next_action_V, next_action_Q
    
    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        node.N += 1
        path = self._select(node)
        while not self._is_terminal_with_depth_limit(path[-1]):
            self._expand_and_evaluate(path[-1])
            # ### debug mode
            # if path[-1].parent is not None:
            #     self._back_propagate(path)
            if self._is_terminal_with_depth_limit(path[-1]) or len(path[-1].children) == 0:
                break
            node = self._puct_select(path[-1])
            path.append(node)
        self._back_propagate(path)
        return path

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.is_terminal or (node.depth - self.root.depth) >= self.depth_limit

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        path = []
        while True:
            path.append(node)
            if node.children is None or len(node.children) == 0 or self._is_terminal_with_depth_limit(node):
                return path
            node = self._puct_select(node)

    def _puct(self, node: MCTSNode) -> float:

        return node.Q + self.w_exp * node.p * np.sqrt(node.parent.N) / (1 + node.N)

    def _uct(self, node: MCTSNode) -> float:
        # "used in ML-Master setting"
        node.p = 1 
        c = 1
        return node.Q / node.N + c * np.sqrt( np.log(node.parent.N) /  node.N)

    def _puct_select(self, node: MCTSNode) -> MCTSNode:
        xnode = max(node.children, key=self._puct)
        return xnode

    def _expand_and_evaluate(self, node: MCTSNode):
        if node.state is None:
            node.state = self.world_model.step(node.parent.state, node.action, node.log_probs)
            node.is_terminal = self.world_model.is_terminal(node.state)
        
        if node.is_terminal:
            return
        node.state = self.get_current_selection()

        actions = self.search_config.get_actions(self.policy_model, node.state, add_kl=self.add_kl)
        
        action_batch, log_probs_batch, ref_log_probs_batch = [], [], []
        for action, (log_probs, ref_log_probs), _ in actions:
            action_batch.append(action)
            log_probs_batch.append(log_probs)
            ref_log_probs_batch.append(ref_log_probs)

        reward_value_batch = self.search_config.get_values(self.policy_model, node.state, action_batch, 
                                                           log_probs_batch, ref_log_probs_batch, 
                                                           add_kl=self.add_kl, parent_depth=node.depth,
                                                           parent_value=node.value)

        children = []
        for (action, (log_probs, ref_log_probs), embs), (value, base_rewards, is_terminal) in zip(actions, reward_value_batch):
            child = MCTSNode(state=None, action=action, parent=node, 
                             base_rewards=base_rewards, value=value, 
                             embeddings=embs, log_probs=log_probs, ref_log_probs=ref_log_probs,
                             is_terminal=is_terminal, length_penalty=self.length_penalty)
            children.append(child)
        node.children = children if node.children is None else node.children + children

    def _back_propagate(self, path: list[MCTSNode]):
        node = path[-1]
        node.Q = node.r + self.gamma * node.V
        node.N += 1
        for node in reversed(path[:-1]):
            node.V = sum(max(1, child.N) * child.Q for child in node.children) / sum(max(1, child.N) for child in node.children)
            node.N += 1
            if node.action is not None:
                node.Q = node.r + self.gamma * node.V

    def search(self):
        if self.root is None:
            self.root = MCTSNode(state=self.world_model.init_state(), action=None, parent=None, length_penalty=self.length_penalty)
        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []

        n_iters = self.n_iters if self.root.depth else self.n_iters * 4     # iterate more at the starting point
        for _ in range(n_iters):
            path = self.iterate(self.root)
            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(deepcopy(path))


    def get_sota_hypothesis_and_experiment(self) -> tuple[Hypothesis | None, Experiment | None]:
        """Access the last experiment result, sub-task, and the corresponding hypothesis."""
        # TODO: The return value does not align with the signature.
        for experiment, feedback in self.hist[::-1]:
            if feedback.decision:
                return experiment.hypothesis, experiment

        return None, None

    def is_selection_new_tree(self, selection: tuple[int, ...] | None = None) -> bool:
        """
        Check if the current trace is a new tree.
        - selection maybe (-1,) when the dag_parent is empty.
        """
        if selection is None:
            selection = self.get_current_selection()

        return selection == self.NEW_ROOT or len(self.dag_parent) == 0

    def get_current_selection(self) -> tuple[int, ...]:
        return self.current_selection

    def set_current_selection(self, selection: tuple[int, ...]) -> None:
        self.current_selection = selection

    def get_parent_exps(
        self,
        selection: tuple[int, ...] | None = None,
    ) -> list[Trace.NodeType]:
        """
        Collect all ancestors of the given selection.
        The return list follows the order of [root->...->parent->current_node].
        """
        if selection is None:
            selection = self.get_current_selection()

        if self.is_selection_new_tree(selection):
            return []

        return [self.hist[i] for i in self.get_parents(selection[0])]

    def exp2idx(self, exp: Experiment | list[Experiment]) -> int | list[int] | None:
        if isinstance(exp, list):
            exps: list[Experiment] = exp

            # keep the order
            exp_to_index: dict[Experiment, int] = {_exp: i for i, (_exp, _) in enumerate(self.hist)}
            return [exp_to_index[_exp] for _exp in exps]
        for i, (_exp, _) in enumerate(self.hist):
            if _exp == exp:
                return i
        return None

    def idx2exp(self, idx: int | list[int]) -> Experiment | list[Experiment]:
        if isinstance(idx, list):
            idxs: list[int] = idx
            return [self.hist[_idx][0] for _idx in idxs]
        return self.hist[idx][0]

    def is_parent(self, parent_idx: int, child_idx: int) -> bool:
        ancestors = self.get_parents(child_idx)
        return parent_idx in ancestors

    def get_parents(self, child_idx: int) -> list[int]:
        if self.is_selection_new_tree((child_idx,)):
            return []

        ancestors: list[int] = []
        curr = child_idx
        while True:
            ancestors.insert(0, curr)
            parent_tuple = self.dag_parent[curr]
            if not parent_tuple or parent_tuple[0] == curr:
                break
            curr = parent_tuple[0]
        return ancestors
