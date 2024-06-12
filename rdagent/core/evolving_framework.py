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


class QlibEvolvableSubjects(EvolvableSubjects):
    ...


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


class EvoAgent(ABC):
    def __init__(self, max_loop, evolving_strategy) -> None:
        self.max_loop = max_loop
        self.evolving_strategy = evolving_strategy
    
    @abstractmethod
    def multistep_evolve(self, evo: EvolvableSubjects, eva: Evaluator | Feedback, **kwargs: Any) -> EvolvableSubjects:
        pass


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
