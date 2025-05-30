# TODO: remove `self.scen` if traces will be passed into the instance.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from rdagent.core.evaluation import Feedback
from rdagent.core.experiment import ASpecificExp, Experiment
from rdagent.core.knowledge_base import KnowledgeBase
from rdagent.core.scenario import Scenario

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
Reason: {self.reason}"""

    # source: data_ana | model_nan = None


# Origin(path of repo/data/feedback) => view/summarization => generated Hypothesis


class ExperimentFeedback(Feedback):
    def __init__(
        self,
        reason: str,
        *,
        decision: bool,
        exception: Exception | None = None,
    ) -> None:
        self.decision = decision
        self.reason = reason
        # Exception is not None means failing to generate runnable experiments due to exception.
        # Runable reuslts are not always good.
        self.exception: Exception | None = (
            exception  # if the experiment raises exception, it will be integrated into part of the feedback.
        )

    def __bool__(self) -> bool:
        return self.decision

    def __str__(self) -> str:
        return f"Decision: {self.decision}\nReason: {self.reason}"

    @classmethod
    def from_exception(cls, e: Exception) -> ExperimentFeedback:
        """
        A convenient method to create Feedback from an exception.
        """
        return cls(decision=False, reason=f"The experiment fails due to {e!s}", exception=e)


class HypothesisFeedback(ExperimentFeedback):
    def __init__(
        self,
        observations: str,
        hypothesis_evaluation: str,
        new_hypothesis: str,
        reason: str,
        *,
        decision: bool,
    ) -> None:
        super().__init__(reason, decision=decision)
        self.observations = observations
        self.hypothesis_evaluation = hypothesis_evaluation
        self.new_hypothesis = new_hypothesis

    def __str__(self) -> str:
        return f"""{super().__str__()}
Observations: {self.observations}
Hypothesis Evaluation: {self.hypothesis_evaluation}
New Hypothesis: {self.new_hypothesis}"""


ASpecificScen = TypeVar("ASpecificScen", bound=Scenario)
ASpecificKB = TypeVar("ASpecificKB", bound=KnowledgeBase)


class Trace(Generic[ASpecificScen, ASpecificKB]):
    NodeType = tuple[Experiment, ExperimentFeedback]  # Define NodeType as a new type representing the tuple

    def __init__(self, scen: ASpecificScen, knowledge_base: ASpecificKB | None = None) -> None:
        self.scen: ASpecificScen = scen
        self.hist: list[Trace.NodeType] = (
            []
        )  # List of tuples containing experiments and their feedback, organized over time.
        self.dag_parent: list[tuple[int, ...]] = []  # List of tuples representing parent indices in the DAG structure.
        # (,) represents no parent; (1,) presents one parent; (1, 2) represents two parents.

        # TODO: self.hist is 2-tuple now, remove hypothesis from it, change old code for this later.
        self.knowledge_base: ASpecificKB | None = knowledge_base

    def get_sota_hypothesis_and_experiment(self) -> tuple[Hypothesis | None, Experiment | None]:
        """Access the last experiment result, sub-task, and the corresponding hypothesis."""
        # TODO: The return value does not align with the signature.
        for experiment, feedback in self.hist[::-1]:
            if feedback.decision:
                return experiment.hypothesis, experiment

        return None, None


class CheckpointSelector:
    """
    In the trace, we may start from any check point (we'll represent it as a variable `from_checkpoint_idx`)
    """

    @abstractmethod
    def get_selection(self, trace: Trace) -> tuple[int, ...] | None:
        """
        checkpoint_idx represents the place where we want to create a new node.
        the return value should be the idx of target node (the parent of the new generating node).
        - `(-1, )` represents starting from the latest trial in the trace - default value
        - `(idx, )` represents starting from the `idx`-th trial in the trace.
        - `None` represents starting from scratch (start a new trace)

        - More advanced selection strategies in `select.py`
        """


class SOTAexpSelector:
    """
    Select the SOTA experiment from the trace to submit
    """

    @abstractmethod
    def get_sota_exp_to_submit(self, trace: Trace) -> Experiment | None:
        """
        Select the SOTA experiment from the trace to submit
        """


class ExpGen(ABC):

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def gen(self, trace: Trace) -> Experiment:
        """
        Generate the experiment based on the trace.

        `ExpGen().gen()` play a role like

        .. code-block:: python

            # ExpGen().gen() ==
            Hypothesis2Experiment().convert(
                HypothesisGen().gen(trace)
            )
        """


class HypothesisGen(ABC):

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


class Experiment2Feedback(ABC):
    """ "Generated feedbacks on the hypothesis from **Executed** Implementations of different tasks
    & their comparisons with previous performances"""

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def generate_feedback(self, exp: Experiment, trace: Trace) -> ExperimentFeedback:
        """
        The `exp` should be executed and the results should be included, as well as the comparison
        between previous results (done by LLM).
        For example: `mlflow` of Qlib will be included.
        """
        error_message = "generate_feedback method is not implemented."
        raise NotImplementedError(error_message)
