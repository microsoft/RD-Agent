"""

"""

from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Tuple, TypeVar

from rdagent.core.evolving_framework import Feedback
from rdagent.core.experiment import Experiment, Implementation, Loader, Task

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


class Scenario(ABC):

    @property
    @abstractmethod
    def background(self):
        """Background information"""

    @property
    @abstractmethod
    def source_data(self):
        """Source data description"""

    @property
    @abstractmethod
    def interface(self):
        """Interface description about how to run the code"""

    @property
    @abstractmethod
    def simulator(self):
        """Simulator description"""

    @abstractmethod
    def get_scenario_all_desc(self) -> str:
        """Combine all the description together"""


class HypothesisFeedback(Feedback):
    def __init__(self, observations: str, feedback_for_hypothesis: str, new_hypothesis: str, reasoning: str, attitude: str):
        self.observations = observations
        self.feedback_for_hypothesis = feedback_for_hypothesis
        self.new_hypothesis = new_hypothesis
        self.reasoning = reasoning
        self.attitude = attitude

    def __str__(self):
        return (f"HypothesisFeedback(\n"
                f"  Observations: {self.observations},\n"
                f"  Feedback for Hypothesis: {self.feedback_for_hypothesis},\n"
                f"  New Hypothesis: {self.new_hypothesis},\n"
                f"  Reasoning: {self.reasoning}\n"
                f")")


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
