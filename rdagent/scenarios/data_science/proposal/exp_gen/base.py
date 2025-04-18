from abc import abstractmethod
from typing import Literal

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.evolving_framework import KnowledgeBase
from rdagent.core.proposal import ExperimentFeedback, Hypothesis, Trace
from rdagent.scenarios.data_science.experiment.experiment import COMPONENT, DSExperiment
from rdagent.scenarios.data_science.scen import DataScienceScen


class DSHypothesis(Hypothesis):
    def __init__(
        self,
        component: COMPONENT,
        hypothesis: str = "",
        reason: str = "",
        concise_reason: str = "",
        concise_observation: str = "",
        concise_justification: str = "",
        concise_knowledge: str = "",
        problem_name: str = "",
        problem_desc: str = "",
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
        if self.hypothesis == "":
            return f"No hypothesis available. Trying to construct the first runnable {self.component} component."
        lines = []
        if self.problem_name is not None and self.problem_desc is not None:
            lines.append(f"Target Problem name: {self.problem_name}")
            lines.append(f"Target Problem: {self.problem_desc}")
        lines.extend(
            [f"Chosen Component: {self.component}", f"Hypothesis: {self.hypothesis}", f"Reason: {self.reason}"]
        )
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

    COMPLETE_ORDER = ("DataLoadSpec", "FeatureEng", "Model", "Ensemble", "Workflow")

    def get_current_selection(self) -> tuple[int, ...]:
        return self.current_selection

    def set_current_selection(self, selection: tuple[int, ...]) -> None:
        self.current_selection = selection

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
        self, search_type: Literal["all", "ancestors"] = "ancestors"
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

        selection = self.get_current_selection()
        if selection is None:
            # selection is None, which means we switch to a new trace, which is not implemented yet
            return []

        return self.collect_all_ancestors(selection) if search_type == "ancestors" else self.hist

    def collect_all_ancestors(
        self,
        selection: tuple[int, ...] = (-1,),
    ) -> list[tuple[DSExperiment, ExperimentFeedback]]:
        """
        Collect all ancestors of the given selection.
        The return list follows the order of [root->...->parent->current_node].
        """

        if len(self.dag_parent) == 0:
            return []

        else:
            all_ancestors = []

            # start from the latest selection
            current_node_idx = selection[0]

            # add the current node to the list
            all_ancestors.insert(0, self.hist[current_node_idx])

            parent_idx = self.dag_parent[current_node_idx]

            while len(parent_idx) > 0:
                all_ancestors.insert(0, self.hist[parent_idx[0]])
                parent_idx = self.dag_parent[parent_idx[0]]

        return all_ancestors

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
    ) -> list[tuple[DSExperiment, ExperimentFeedback]]:
        """
        Retrieve a list of experiments and feedbacks based on the return_type.
        """
        search_list = self.retrieve_search_list(search_type)

        final_component = self.COMPLETE_ORDER[-1]
        has_final_component = True if DS_RD_SETTING.coder_on_whole_pipeline else False
        exp_and_feedback_list = []
        for exp, fb in search_list:
            if has_final_component:
                if return_type == "all":
                    exp_and_feedback_list.append((exp, fb))
                elif return_type == "failed" and not fb.decision:
                    exp_and_feedback_list.append((exp, fb))
                elif return_type == "sota" and fb.decision:
                    exp_and_feedback_list.append((exp, fb))
            if exp.hypothesis.component == final_component and fb:
                has_final_component = True
        return exp_and_feedback_list

    def sota_experiment(
        self,
        search_type: Literal["all", "ancestors"] = "ancestors",
    ) -> DSExperiment | None:
        """

        Returns
        -------
        Experiment or None
            The experiment result if found, otherwise None.
        """
        search_list = self.retrieve_search_list(search_type)

        if DS_RD_SETTING.coder_on_whole_pipeline or self.next_incomplete_component() is None:
            for exp, ef in search_list[::-1]:
                # the sota exp should be accepted decision and all required components are completed.
                if ef.decision:
                    return exp
        return None

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
        search_list = self.retrieve_search_list(search_type)

        for exp, ef in search_list[::-1]:
            return exp
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
