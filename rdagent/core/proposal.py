"""

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from rdagent.core.evaluation import Feedback
from rdagent.core.experiment import ASpecificExp, Experiment
from rdagent.core.knowledge_base import KnowledgeBase
from rdagent.core.scenario import Scenario

if TYPE_CHECKING:
    from rdagent.core.prompts import Prompts

# class data_ana: XXX


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
                Reason: {self.reason}
                Concise Reason & Knowledge: {self.concise_reason}
                Concise Observation: {self.concise_observation}
                Concise Justification: {self.concise_justification}
                Concise Knowledge: {self.concise_knowledge}
                """

    # source: data_ana | model_nan = None


# Origin(path of repo/data/feedback) => view/summarization => generated Hypothesis


class HypothesisFeedback(Feedback):
    def __init__(
        self,
        observations: str,
        hypothesis_evaluation: str,
        new_hypothesis: str,
        reason: str,
        decision: bool,
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
ASpecificKB = TypeVar("ASpecificKB", bound=KnowledgeBase)


class Trace(Generic[ASpecificScen, ASpecificKB]):
    def __init__(self, scen: ASpecificScen, knowledge_base: ASpecificKB | None = None) -> None:
        self.scen: ASpecificScen = scen
        self.hist: list[tuple[Hypothesis, Experiment, HypothesisFeedback]] = []
        self.knowledge_base: ASpecificKB | None = knowledge_base

    def get_sota_hypothesis_and_experiment(self) -> tuple[Hypothesis | None, Experiment | None]:
        """Access the last experiment result, sub-task, and the corresponding hypothesis."""
        # TODO: The return value does not align with the signature.
        for hypothesis, experiment, feedback in self.hist[::-1]:
            if feedback.decision:
                return hypothesis, experiment

        return None, None


class HypothesisGen(ABC):
    # NOTE: the design is a little wierd
    # - Sometimes we want accurate access the prompts in a specific level
    #   - It renders the prompt to a specific abstract level
    # - Sometimes we want to access the most recent level prompts
    prompts: Prompts  # this is a class level prompt.

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
