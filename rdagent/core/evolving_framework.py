from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class Feedback:
    pass


class Knowledge:
    pass


class QueriedKnowledge:
    pass


class KnowledgeBase(ABC):
    @abstractmethod
    def query(
        self,
    ) -> QueriedKnowledge | None:
        raise NotImplementedError


class EvolvableSubjects:
    """The target object to be evolved"""

    def clone(self) -> EvolvableSubjects:
        return copy.deepcopy(self)


class QlibEvolvableSubjects(EvolvableSubjects): ...


class Evaluator(ABC):
    """Both external EvolvableSubjects and internal evovler, it is

    FAQ:
    - Q: If we have a external whitebox evaluator, do we need a
         intenral EvolvableSubjects?
      A: When the external evovler is very complex, maybe a internal LLM-based evovler
      may provide more understandable feedbacks.
    """

    @abstractmethod
    def evaluate(self, evo: EvolvableSubjects, **kwargs: Any) -> Feedback:
        raise NotImplementedError


class SelfEvaluator(Evaluator):
    pass


@dataclass
class EvoStep:
    """At a specific step,
    based on
    - previous trace
    - newly RAG kownledge `QueriedKnowledge`

    the EvolvableSubjects is evolved to a new one `EvolvableSubjects`.

    (optional) After evaluation, we get feedback `feedback`.
    """

    evolvable_subjects: EvolvableSubjects
    queried_knowledge: QueriedKnowledge | None = None
    feedback: Feedback | None = None


class EvolvingStrategy(ABC):
    @abstractmethod
    def evolve(
        self,
        *evo: EvolvableSubjects,
        evolving_trace: list[EvoStep] | None = None,
        queried_knowledge: QueriedKnowledge | None = None,
        **kwargs: Any,
    ) -> EvolvableSubjects:
        """The evolving trace is a list of (evolvable_subjects, feedback) ordered
        according to the time.

        The reason why the parameter is important for the evolving.
        - evolving_trace: the historical feedback is important.
        - queried_knowledge: queried knowledge
        """


class RAGStrategy(ABC):
    """Retrival Augmentation Generation Strategy"""

    def __init__(self, knowledgebase: KnowledgeBase) -> None:
        self.knowledgebase = knowledgebase

    @abstractmethod
    def query(
        self,
        evo: EvolvableSubjects,
        evolving_trace: list[EvoStep],
        **kwargs: Any,
    ) -> QueriedKnowledge | None:
        pass

    @abstractmethod
    def generate_knowledge(
        self,
        evolving_trace: list[EvoStep],
        *,
        return_knowledge: bool = False,
        **kwargs: Any,
    ) -> Knowledge | None:
        """Generating new knowledge based on the evolving trace.
        - It is encouraged to query related knowledge before generating new knowledge.

        RAGStrategy should maintain the new knowledge all by itself.
        """


class EvoAgent:
    """It is responsible for driving the workflow."""

    evolving_trace: list[EvoStep]

    def __init__(
        self,
        evolving_strategy: EvolvingStrategy,
        rag: RAGStrategy | None = None,
    ) -> None:
        self.evolving_trace = []
        self.evolving_strategy = evolving_strategy
        self.rag = rag

    def step_evolving(
        self,
        evo: EvolvableSubjects,
        eva: Evaluator | Feedback,
        *,
        with_knowledge: bool = False,
        with_feedback: bool = True,
        knowledge_self_gen: bool = False,
    ) -> EvolvableSubjects:
        """Common evolving mode are supported in this api .
        - Interactive evolving:
            - `with_feedback=True` and `eva` is a external Evaluator.

        - Knowledge-driven evolving:
            - `with_knowledge=True` and related knowledge are
            queried based on `self.rag`

        - Self-evolving: we have two ways to self-evolve.
            - 1) self generating knowledge and then evolve
                - `knowledge_self_gen=True` and `with_knowledge=True`
            - 2) self evaluate to generate feedback and then evolve
                - `with_feedback=True` and `eva` is a internal Evaluator.
        """
        # knowledge self-evolving
        if knowledge_self_gen and self.rag is not None:
            self.rag.generate_knowledge(self.evolving_trace)

        # RAG
        queried_knowledge = None
        if with_knowledge and self.rag is not None:
            queried_knowledge = self.rag.query(evo, self.evolving_trace)

        # Evolve
        evo = self.evolving_strategy.evolve(
            evo=evo,
            evolving_trace=self.evolving_trace,
            queried_knowledge=queried_knowledge,
        )
        es = EvoStep(evo, queried_knowledge)

        # Evaluate
        if with_feedback:
            es.feedback = eva if isinstance(eva, Feedback) else eva.evaluate(evo, queried_knowledge=queried_knowledge)

        # Update trace
        self.evolving_trace.append(es)
        return evo
