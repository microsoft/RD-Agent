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


class HypothesisFeedback(Feedback): ...


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
