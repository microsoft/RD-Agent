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
        lines = []
        if self.problem is not None:
            lines.append(f"Target Problem: {self.problem}")
        lines.extend(
            [f"Chosen Component: {self.component}", f"Hypothesis: {self.hypothesis}", f"Reason: {self.reason}"]
        )
        return "\n".join(lines)


class DSTrace(Trace[DataScienceScen, KnowledgeBase]):
    def __init__(self, scen: DataScienceScen, knowledge_base: KnowledgeBase | None = None) -> None:
        self.scen: DataScienceScen = scen
        self.hist: list[tuple[DSExperiment, ExperimentFeedback]] = []
        self.knowledge_base = knowledge_base

    COMPLETE_ORDER = ("DataLoadSpec", "FeatureEng", "Model", "Ensemble", "Workflow")

    def next_incomplete_component(self) -> COMPONENT | None:
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
        has_final_component = True if DS_RD_SETTING.coder_on_whole_pipeline else False
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
        if DS_RD_SETTING.coder_on_whole_pipeline or self.next_incomplete_component() is None:
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
