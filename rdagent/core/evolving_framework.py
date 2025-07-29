from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from rdagent.core.evaluation import EvaluableObj
from rdagent.core.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from rdagent.core.evaluation import Feedback
    from rdagent.core.scenario import Scenario


class Knowledge:
    pass


class QueriedKnowledge:
    pass


class EvolvingKnowledgeBase(KnowledgeBase):
    @abstractmethod
    def query(
        self,
    ) -> QueriedKnowledge | None:
        raise NotImplementedError


class EvolvableSubjects(EvaluableObj):
    """The target object to be evolved"""

    def clone(self) -> EvolvableSubjects:
        return copy.deepcopy(self)


ASpecificEvolvableSubjects = TypeVar("ASpecificEvolvableSubjects", bound=EvolvableSubjects)


@dataclass
class EvoStep(Generic[ASpecificEvolvableSubjects]):
    """At a specific step,
    based on
    - previous trace
    - newly RAG knowledge `QueriedKnowledge`

    the EvolvableSubjects is evolved to a new one `EvolvableSubjects`.

    (optional) After evaluation, we get feedback `feedback`.
    """

    evolvable_subjects: ASpecificEvolvableSubjects

    queried_knowledge: QueriedKnowledge | None = None
    feedback: Feedback | None = None


class EvolvingStrategy(ABC, Generic[ASpecificEvolvableSubjects]):
    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def evolve(
        self,
        *evo: ASpecificEvolvableSubjects,
        evolving_trace: list[EvoStep[ASpecificEvolvableSubjects]] | None = None,
        queried_knowledge: QueriedKnowledge | None = None,
        **kwargs: Any,
    ) -> ASpecificEvolvableSubjects:
        """The evolving trace is a list of (evolvable_subjects, feedback) ordered
        according to the time.

        The reason why the parameter is important for the evolving.
        - evolving_trace: the historical feedback is important.
        - queried_knowledge: queried knowledge
        """


class RAGStrategy(ABC, Generic[ASpecificEvolvableSubjects]):
    """Retrieval Augmentation Generation Strategy"""

    def __init__(self, knowledgebase: EvolvingKnowledgeBase) -> None:
        self.knowledgebase: EvolvingKnowledgeBase = knowledgebase

    @abstractmethod
    def query(
        self,
        evo: ASpecificEvolvableSubjects,
        evolving_trace: list[EvoStep],
        **kwargs: Any,
    ) -> QueriedKnowledge | None:
        pass

    @abstractmethod
    def generate_knowledge(
        self,
        evolving_trace: list[EvoStep[ASpecificEvolvableSubjects]],
        *,
        return_knowledge: bool = False,
        **kwargs: Any,
    ) -> Knowledge | None:
        """Generating new knowledge based on the evolving trace.
        - It is encouraged to query related knowledge before generating new knowledge.

        RAGStrategy should maintain the new knowledge all by itself.
        """
