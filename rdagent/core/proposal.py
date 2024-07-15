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
    
    def __str__(self) -> str:
        return f"""Hypothesis: {self.hypothesis}
Reason: {self.reason}"""

    # source: data_ana | model_nan = None


# Origin(path of repo/data/feedback) => view/summarization => generated Hypothesis


class HypothesisFeedback(Feedback):
    def __init__(self, observations: str, hypothesis_evaluation: str, new_hypothesis: str, reason: str, decision: bool):
        self.observations = observations
        self.hypothesis_evaluation = hypothesis_evaluation
        self.new_hypothesis = new_hypothesis
        self.reason = reason
        self.decision = decision

    def __bool__(self):
        return self.decision


ASpecificScen = TypeVar("ASpecificScen", bound=Scenario)


class Trace(Generic[ASpecificScen]):
    def __init__(self, scen: ASpecificScen) -> None:
        self.scen: ASpecificScen = scen
        self.hist: list[Tuple[Hypothesis, Experiment, HypothesisFeedback]] = []

    def get_last_experiment_info(self) -> Tuple[Hypothesis, ASpecificTask, Any]:
        """Access the last experiment result, sub-task, and the corresponding hypothesis."""
        # TODO: The return value does not align with the signature.
        if not self.hist:
            return None
        last_hypothesis, last_experiment, _ = self.hist[-1]
        last_task = last_experiment.sub_tasks[-1]
        last_result = last_experiment.result
        return last_hypothesis, last_task, last_result

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


ASpecificExp = TypeVar("ASpecificExp", bound=Experiment)


class Hypothesis2Experiment(ABC, Generic[ASpecificExp]):
    """
    [Abstract description => concrete description] => Code implement
    """

    @abstractmethod
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> ASpecificExp:
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
        raise NotImplementedError("generateFeedback method is not implemented.")

    # def generateResultComparison()
