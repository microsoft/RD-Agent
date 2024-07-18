"""

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from rdagent.core.evaluation import Feedback
from rdagent.core.experiment import ASpecificExp, Experiment
from rdagent.core.scenario import Scenario

# class data_ana: XXX


class Hypothesis:
    """
    TODO: We may have better name for it.

    Name Candidates:
    - Belief
    """

    def __init__(self, hypothesis: str, reason: str) -> None:
        self.hypothesis: str = hypothesis
        self.reason: str = reason

    def __str__(self) -> str:
        return f"""Hypothesis: {self.hypothesis}
Reason: {self.reason}"""

    # source: data_ana | model_nan = None


# Origin(path of repo/data/feedback) => view/summarization => generated Hypothesis


class HypothesisFeedback(Feedback):
    def __init__(
        self,
        observations: str,
        hypothesis_evaluation: str,
        new_hypothesis: str,
        reason: str,
        decision: bool,  # noqa: FBT001
    ) -> None:
        self.observations = observations
        self.hypothesis_evaluation = hypothesis_evaluation
        self.new_hypothesis = new_hypothesis
        self.reason = reason
        self.decision = decision

    def __bool__(self) -> bool:
        return self.decision

    def __str__(self) -> str:
        return f"""Observations: {self.observations}
Hypothesis Evaluation: {self.hypothesis_evaluation}
New Hypothesis: {self.new_hypothesis}
Decision: {self.decision}
Reason: {self.reason}"""


ASpecificScen = TypeVar("ASpecificScen", bound=Scenario)


class Trace(Generic[ASpecificScen]):
    def __init__(self, scen: ASpecificScen) -> None:
        self.scen: ASpecificScen = scen
        self.hist: list[tuple[Hypothesis, Experiment, HypothesisFeedback]] = []

    def get_sota_hypothesis_and_experiment(self) -> tuple[Hypothesis, Experiment]:
        """Access the last experiment result, sub-task, and the corresponding hypothesis."""
        # TODO: The return value does not align with the signature.
        for hypothesis, experiment, feedback in self.hist[::-1]:
            if feedback.decision:
                return hypothesis, experiment

        return None, None


class HypothesisGen(ABC):
    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def gen(self, trace: Trace) -> Hypothesis:
        # def gen(self, scenario_desc: str, ) -> Hypothesis:
        """
        Motivation of the variable `scenario_desc`:
        - Mocking a data-scientist is observing the scenario.

        scenario_desc may conclude:
        - data observation:
            - Original or derivative
        - Task information:
        """


class Hypothesis2Experiment(ABC, Generic[ASpecificExp]):
    """
    [Abstract description => concrete description] => Code implement
    """

    @abstractmethod
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> ASpecificExp:
        """Connect the idea proposal to implementation"""
        ...


# Boolean, Reason, Confidence, etc.


class HypothesisExperiment2Feedback(ABC):
    """ "Generated feedbacks on the hypothesis from **Executed** Implementations of different tasks
    & their comparisons with previous performances"""

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        The `exp` should be executed and the results should be included, as well as the comparison
        between previous results (done by LLM).
        For example: `mlflow` of Qlib will be included.
        """
        error_message = "generate_feedback method is not implemented."
        raise NotImplementedError(error_message)
