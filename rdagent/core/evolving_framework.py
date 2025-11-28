from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator, Generic, TypeVar

from rdagent.core.evaluation import EvaluableObj, Evaluator
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

    def evolve_iter(self,
                    evo: ASpecificEvolvableSubjects,
                    queried_knowledge: QueriedKnowledge = None,
                    evolving_trace: list[EvoStep] = []) -> Generator[ASpecificEvolvableSubjects, None, None]:
        """
        The evolving trace is a list of (evolvable_subjects, feedback) ordered
        according to the time.

        The reason why the parameter is important for the evolving.
        - evolving_trace: the historical feedback is important.
        - queried_knowledge: queried knowledge

        Assumptions:
        - The evolving process will make modifications in-place. So the yield evo and the parameter evo are the same object!!!!


        Typical implementation of this method is:

        .. code-block:: python

            for evolve_function in self.evolve_func_iter():
                yield evolve_function(evo=evo, queried_knowledge=queried_knowledge, evolving_trace=evolving_trace)
                # evolve_function will return a partial evolved solution.
        """


class IterEvaluator(Evaluator):
    """
    Some evolving implementation (i.e. evolve_iter) will iteratively implement partial solutions before a complete final solution.

    According to that strategy, we have iterative evaluation
    """

    @abstractmethod
    def evaluate_iter(self) -> Generator[Feedback, EvaluableObj | None, Feedback]:
        """

        1) It will yield a evaluation for each implement part and yield the feedback for that part.
        2) And finally, it will get the summarize all the feedback and return a overall feedback.

        Sending a None feedback will stop the evaluation chain and just return the overall feedback.

        A typical implementation of this method is:

        .. code-block:: python

            evo = yield Feedback()  # it will receive the evo first, so the first yield is for get the sent evo instead of generate useful feedback
            assert evo is not None
            for partial_eval_func in self.evaluate_func_iter():
                partial_fb = partial_eval_func(evo)
                # return the partial feedback and receive the evolved solution for next iteration
                evo_next_iter = yield partial_fb
                evo = evo_next_iter

            final_fb = get_final_fb(...)
            return final_fb

        """


class RAGStrategy(ABC, Generic[ASpecificEvolvableSubjects]):
    """Retrieval Augmentation Generation Strategy"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.knowledgebase: EvolvingKnowledgeBase = self.load_or_init_knowledge_base(*args, **kwargs)

    @abstractmethod
    def load_or_init_knowledge_base(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> EvolvingKnowledgeBase:
        pass

    @abstractmethod
    def query(
        self,
        evo: ASpecificEvolvableSubjects,
        evolving_trace: list[EvoStep],
        **kwargs: Any,
    ) -> QueriedKnowledge:
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

    @abstractmethod
    def dump_knowledge_base(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def load_dumped_knowledge_base(self, *args: Any, **kwargs: Any) -> None:
        """This is to load the dumped knowledge base.
        It's mainly used in parallel coding of which several coder shares the same knowledge base.
        Then the agent should load the knowledge base from others before updating it.
        """
