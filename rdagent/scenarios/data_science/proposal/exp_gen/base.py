from abc import abstractmethod
from typing import Literal

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
        problem: str = "",
    ) -> None:
        super().__init__(
            hypothesis, reason, concise_reason, concise_observation, concise_justification, concise_knowledge
        )
        self.component = component
        self.problem = problem

    def __str__(self) -> str:
        if self.hypothesis == "":
            return f"No hypothesis available. Trying to construct the first runnable {self.component} component."
        return f"""Chosen Component: {self.component}
Hypothesis: {self.hypothesis}
Reason: {self.reason}
Concise Reason & Knowledge: {self.concise_reason}
Concise Observation: {self.concise_observation}
Concise Justification: {self.concise_justification}
Concise Knowledge: {self.concise_knowledge}
"""


class DSTrace(Trace[DataScienceScen, KnowledgeBase]):
    hist: list[tuple[DSExperiment, ExperimentFeedback]]

    COMPLETE_ORDER = ("DataLoadSpec", "FeatureEng", "Model", "Ensemble", "Workflow")

    # TODO: change the logic based on current selection
    def next_incomplete_component(self, selection: tuple[int, ...] = ()) -> COMPONENT | None:
        """
        NOTE:
        - A component will be complete until get True decision feedback !!!
        """
        for c in self.COMPLETE_ORDER:
            if not self.has_component(c):
                return c
        return None

    def has_component(self, component: COMPONENT) -> bool:
        for exp, fb in self.hist:
            assert isinstance(exp.hypothesis, DSHypothesis), "Hypothesis should be DSHypothesis (and not None)"
            if exp.hypothesis.component == component and fb:
                return True
        return False

    def experiment_and_feedback_list_after_init(
        self, return_type: Literal["sota", "failed", "all"]
    ) -> list[tuple[DSExperiment, ExperimentFeedback]]:
        """
        Retrieve a list of experiments and feedbacks based on the return_type.

        Parameters
        ----------
        return_type : str
            One of "sota", "failed", "all".

        Returns
        -------
        list[tuple[DSExperiment, ExperimentFeedback]]
            List of experiments and feedbacks.
        """

        final_component = self.COMPLETE_ORDER[-1]
        has_final_component = False
        exp_and_feedback_list = []
        for exp, fb in self.hist:
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

    def sota_experiment(self) -> DSExperiment | None:
        """
        Returns
        -------
        Experiment or None
            The experiment result if found, otherwise None.
        """
        if self.next_incomplete_component() is None:
            for exp, ef in self.hist[::-1]:
                # the sota exp should be accepted decision and all required components are completed.
                if ef.decision:
                    return exp
        return None

    def last_successful_exp(self) -> DSExperiment | None:
        """
        Access the last successful experiment even part of the components are not completed.
        """
        for exp, ef in self.hist[::-1]:
            if ef.decision:
                return exp
        return None

    def last_runnable_exp_fb(self) -> tuple[DSExperiment, ExperimentFeedback] | None:
        """
        Access the last runnable experiment (no exception, usually not all task failed) and feedback
        """
        for exp, ef in self.hist[::-1]:
            if ef.exception is None:
                return exp, ef
        return None
