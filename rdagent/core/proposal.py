"""

"""

from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Tuple, TypeVar

from rdagent.core.evaluation import Feedback
from rdagent.core.experiment import Experiment
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

    # source: data_ana | model_nan = None


# Origin(path of repo/data/feedback) => view/summarization => generated Hypothesis

class HypothesisFeedback(Feedback):
    def __init__(self, observations: str, feedback_for_hypothesis: str, new_hypothesis: str, reasoning: str, attitude: str):
        self.observations = observations
        self.feedback_for_hypothesis = feedback_for_hypothesis
        self.new_hypothesis = new_hypothesis
        self.reasoning = reasoning
        self.attitude = attitude

    def __repr__(self):
        return (f"HypothesisFeedback(observations={self.observations}, "
                f"feedback_for_hypothesis={self.feedback_for_hypothesis}, "
                f"new_hypothesis={self.new_hypothesis}, "
                f"reasoning={self.reasoning}, "
                f"attitude={self.attitude})")

    def to_dict(self) -> dict:
        return {
            "Observations": self.observations,
            "Feedback for Hypothesis": self.feedback_for_hypothesis,
            "New Hypothesis": self.new_hypothesis,
            "Reasoning": self.reasoning,
            "Attitude": self.attitude
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            observations=data.get("Observations", ""),
            feedback_for_hypothesis=data.get("Feedback for Hypothesis", ""),
            new_hypothesis=data.get("New Hypothesis", ""),
            reasoning=data.get("Reasoning", ""),
            attitude=data.get("Attitude", "no")  # Default to "no" if not provided
        )


ASpecificScen = TypeVar("ASpecificScen", bound=Scenario)


class Trace(Generic[ASpecificScen]):
    def __init__(self, scen: ASpecificScen) -> None:
        self.scen: ASpecificScen = scen
        self.hist: list[Tuple[Hypothesis, Experiment, HypothesisFeedback]] = []


class HypothesisGen:
    def __init__(self, scen: Scenario):
        self.scen = scen

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


class HypothesisSet:
    """
    # drop, append

    hypothesis_imp: list[float] | None  # importance of each hypothesis
    true_hypothesis or false_hypothesis
    """

    def __init__(self, trace: Trace, hypothesis_list: list[Hypothesis] = []) -> None:
        self.hypothesis_list: list[Hypothesis] = hypothesis_list
        self.trace: Trace = trace


ASpecificExp = TypeVar("ASpecificExp", bound=Experiment)

class Hypothesis2Experiment(ABC, Generic[ASpecificExp]):
    """
    [Abstract description => concrete description] => Code implement
    """

    @abstractmethod
    def convert(self, hs: HypothesisSet) -> ASpecificExp:
        """Connect the idea proposal to implementation"""
        ...


# Boolean, Reason, Confidence, etc.

class HypothesisExperiment2Feedback:
    """ "Generated feedbacks on the hypothesis from **Executed** Implementations of different tasks & their comparisons with previous performances"""

    def generateFeedback(self, ti: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        The `ti` should be executed and the results should be included, as well as the comparison between previous results (done by LLM). 
        For example: `mlflow` of Qlib will be included.
        """
        return HypothesisFeedback()

    # def generateResultComparison()
