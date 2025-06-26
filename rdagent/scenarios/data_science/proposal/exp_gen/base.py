from abc import abstractmethod
from typing import List, Literal

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.evolving_framework import KnowledgeBase
from rdagent.core.proposal import ExperimentFeedback, Hypothesis, Trace
from rdagent.scenarios.data_science.experiment.experiment import COMPONENT, DSExperiment
from rdagent.scenarios.data_science.scen import DataScienceScen


class DSHypothesis(Hypothesis):
    def __init__(
        self,
        component: COMPONENT,
        hypothesis: str | None = None,
        reason: str | None = None,
        concise_reason: str | None = None,
        concise_observation: str | None = None,
        concise_justification: str | None = None,
        concise_knowledge: str | None = None,
        problem_name: str | None = None,
        problem_desc: str | None = None,
        problem_label: Literal["SCENARIO_PROBLEM", "FEEDBACK_PROBLEM"] = "FEEDBACK_PROBLEM",
    ) -> None:
        super().__init__(
            hypothesis, reason, concise_reason, concise_observation, concise_justification, concise_knowledge
        )
        self.component = component
        self.problem_name = problem_name
        self.problem_desc = problem_desc
        self.problem_label = problem_label

    def __str__(self) -> str:
        if self.hypothesis is None:
            return f"No hypothesis available. Trying to construct the first runnable {self.component} component."

        lines = []
        if self.problem_name is not None:
            lines.append(f"Target Problem Name: {self.problem_name}")
        if self.problem_desc is not None:
            lines.append(f"Target Problem: {self.problem_desc}")
        lines.append(f"Chosen Component: {self.component}")
        lines.append(f"Hypothesis: {self.hypothesis}")
        if self.reason is not None:
            lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


class DSTrace(Trace[DataScienceScen, KnowledgeBase]):
    def __init__(self, scen: DataScienceScen, knowledge_base: KnowledgeBase | None = None) -> None:
        self.scen: DataScienceScen = scen
        self.hist: list[tuple[DSExperiment, ExperimentFeedback]] = []
        """
        The dag_parent is a list of tuples, each tuple is the parent index of the current node.
        The first element of the tuple is the parent index, the rest are the parent indexes of the parent (not implemented yet).
        If the current node is the root node without parent, the tuple is empty.
        """
        self.dag_parent: list[tuple[int, ...]] = []  # List of tuples representing parent indices in the DAG structure.
        # () represents no parent; (1,) presents one parent; (1, 2) represents two parents.

        self.knowledge_base = knowledge_base
        self.current_selection: tuple[int, ...] = (-1,)

        self.sota_exp_to_submit: DSExperiment | None = None  # grab the global best exp to submit

    COMPLETE_ORDER = ("DataLoadSpec", "FeatureEng", "Model", "Ensemble", "Workflow")

    def set_sota_exp_to_submit(self, exp: DSExperiment) -> None:
        self.sota_exp_to_submit = exp

    @property
    def sub_trace_count(self) -> int:
        return len(self.get_leaves())

    def get_leaves(self) -> list[int, ...]:
        """
        Get the indices of nodes (in hist) that have no children—i.e., "leaves" of current DAG.
        Returns:
            tuple of ints: Indices of leaf nodes.
            - Leaves with lower index comes first.
        """
        # BUG: potential BUG:
        # If we implement the most correct merging logic,  merge 2 traces, will result in a single trace(2 traces currently).
        # So user may get unexpected results when he want to know ho many branches are created.

        # Build a set of all parent indices found in dag_parent (skip empty tuples which represent roots)
        parent_indices = set(idx for parents in self.dag_parent for idx in parents)
        # All node indices
        all_indices = set(range(len(self.hist)))
        # The leaf nodes have no children, so they are not present as parents of any other node
        leaves = list(sorted(all_indices - parent_indices))
        return leaves

    def sync_dag_parent_and_hist(
        self,
    ) -> None:
        """
        Adding corresponding parent index to the dag_parent when the hist is going to be changed.
        Should be called when the hist is changed.
        """

        if len(self.hist) == 0 or len(self.get_current_selection()) == 0:
            # the node we are going to add is the first node of hist / root node of a new sub-trace
            self.dag_parent.append(())

        else:
            current_node_idx = self.current_selection[0]

            if current_node_idx == -1:
                # the current selection is the latest one
                current_node_idx = len(self.hist) - 1

            self.dag_parent.append((current_node_idx,))

    def retrieve_search_list(
        self,
        search_type: Literal["all", "ancestors"] = "ancestors",
        selection: tuple[int, ...] | None = None,
    ) -> list[tuple[DSExperiment, ExperimentFeedback]]:
        """
        Retrieve the search list based on the selection and search_type.

        Parameters
        ----------
        search_type : str
            One of "all", "ancestors".
            - "all": search the whole hist.
            - "ancestors": search the trace from root to the selection.

        Returns
        -------
        list[tuple[DSExperiment, ExperimentFeedback]]
            The search list.
        """
        if search_type == "all":
            return self.hist

        elif search_type == "ancestors":
            return self.get_parent_exps(selection)

        else:
            raise ValueError(f"Invalid search type: {search_type}")

    def next_incomplete_component(
        self,
        search_type: Literal["all", "ancestors"] = "ancestors",
    ) -> COMPONENT | None:
        """
        NOTE:
        - A component will be complete until get True decision feedback !!!

        """
        search_list = self.retrieve_search_list(search_type)

        for c in self.COMPLETE_ORDER:
            """Check if the component is in the ancestors of the selection."""
            if not self.has_component(c, search_list):
                return c

        return None

    def has_component(
        self, component: COMPONENT, search_list: list[tuple[DSExperiment, ExperimentFeedback]] = []
    ) -> bool:
        for exp, fb in search_list:
            assert isinstance(exp.hypothesis, DSHypothesis), "Hypothesis should be DSHypothesis (and not None)"
            if exp.hypothesis.component == component and fb:
                return True
        return False

    def experiment_and_feedback_list_after_init(
        self,
        return_type: Literal["sota", "failed", "all"],
        search_type: Literal["all", "ancestors"] = "all",
        selection: tuple[int, ...] | None = None,
        max_retrieve_num: int | None = None,
    ) -> list[tuple[DSExperiment, ExperimentFeedback]]:
        """
        Retrieve a list of experiments and feedbacks based on the return_type.
        """
        search_list = self.retrieve_search_list(search_type, selection=selection)
        final_component = self.COMPLETE_ORDER[-1]
        has_final_component = True if DS_RD_SETTING.coder_on_whole_pipeline else False
        SOTA_exp_and_feedback_list = []
        failed_exp_and_feedback_list_after_sota = []
        for exp, fb in search_list:
            if has_final_component:
                if fb.decision:
                    SOTA_exp_and_feedback_list.append((exp, fb))
                    failed_exp_and_feedback_list_after_sota = []
                else:
                    failed_exp_and_feedback_list_after_sota.append((exp, fb))
            if exp.hypothesis.component == final_component and fb:
                has_final_component = True
        if max_retrieve_num is not None and (SOTA_exp_and_feedback_list or failed_exp_and_feedback_list_after_sota):
            SOTA_exp_and_feedback_list = SOTA_exp_and_feedback_list[
                -min(max_retrieve_num, len(SOTA_exp_and_feedback_list)) :
            ]
            failed_exp_and_feedback_list_after_sota = failed_exp_and_feedback_list_after_sota[
                -min(max_retrieve_num, len(failed_exp_and_feedback_list_after_sota)) :
            ]
        if return_type == "all":
            return SOTA_exp_and_feedback_list + failed_exp_and_feedback_list_after_sota
        elif return_type == "failed":
            return failed_exp_and_feedback_list_after_sota
        elif return_type == "sota":
            return SOTA_exp_and_feedback_list
        else:
            raise ValueError("Invalid return_type. Must be 'sota', 'failed', or 'all'.")

    def sota_experiment_fb(
        self,
        search_type: Literal["all", "ancestors"] = "ancestors",
        selection: tuple[int, ...] | None = None,
    ) -> tuple[DSExperiment, ExperimentFeedback] | None:
        """
        Returns
        -------
        Experiment or None
            The experiment result if found, otherwise None.
        """
        search_list = self.retrieve_search_list(search_type, selection=selection)

        if DS_RD_SETTING.coder_on_whole_pipeline or self.next_incomplete_component() is None:
            for exp, ef in search_list[::-1]:
                # the sota exp should be accepted decision and all required components are completed.
                if ef.decision:
                    return exp, ef
        return None

    def sota_experiment(
        self,
        search_type: Literal["all", "ancestors"] = "ancestors",
        selection: tuple[int, ...] | None = None,
    ) -> DSExperiment | None:
        res = self.sota_experiment_fb(search_type=search_type, selection=selection)
        if res is not None:
            res = res[0]
        return res

    def last_successful_exp(
        self,
        search_type: Literal["all", "ancestors"] = "ancestors",
    ) -> DSExperiment | None:
        """
        Access the last successful experiment even part of the components are not completed.
        """
        search_list = self.retrieve_search_list(search_type)

        for exp, ef in search_list[::-1]:
            if ef.decision:
                return exp
        return None

    def last_exp(
        self,
        search_type: Literal["all", "ancestors"] = "ancestors",
    ) -> DSExperiment | None:
        """
        Access the last experiment
        """
        if (last_exp_fb := self.last_exp_fb(search_type=search_type)) is not None:
            return last_exp_fb[0]
        return None

    def last_exp_fb(
        self,
        search_type: Literal["all", "ancestors"] = "ancestors",
    ) -> tuple[DSExperiment, ExperimentFeedback] | None:
        """
        Access the last experiment and feedback
        """
        search_list = self.retrieve_search_list(search_type)
        for exp, ef in search_list[::-1]:
            return exp, ef
        return None

    def last_runnable_exp_fb(
        self,
        search_type: Literal["all", "ancestors"] = "ancestors",
    ) -> tuple[DSExperiment, ExperimentFeedback] | None:
        """
        Access the last runnable experiment (no exception, usually not all task failed) and feedback
        """
        search_list = self.retrieve_search_list(search_type)

        for exp, ef in search_list[::-1]:
            if ef.exception is None:
                return exp, ef
        return None
