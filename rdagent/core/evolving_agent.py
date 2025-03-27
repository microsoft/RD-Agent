from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from tqdm import tqdm

if TYPE_CHECKING:
    from rdagent.core.evolving_framework import EvolvableSubjects

from rdagent.core.evaluation import EvaluableObj, Evaluator, Feedback
from rdagent.core.evolving_framework import EvolvingStrategy, EvoStep
from rdagent.log import rdagent_logger as logger

ASpecificEvaluator = TypeVar("ASpecificEvaluator", bound=Evaluator)


class EvoAgent(ABC, Generic[ASpecificEvaluator]):

    def __init__(self, max_loop: int, evolving_strategy: EvolvingStrategy) -> None:
        self.max_loop = max_loop
        self.evolving_strategy = evolving_strategy

    @abstractmethod
    def multistep_evolve(
        self,
        evo: EvolvableSubjects,
        eva: ASpecificEvaluator | Feedback,
    ) -> Generator[EvolvableSubjects, None, None]:
        """
        yield EvolvableSubjects for caller for easier process control and logging.
        """


class RAGEvaluator(Evaluator):

    @abstractmethod
    def evaluate(
        self,
        eo: EvaluableObj,
        queried_knowledge: object = None,
    ) -> Feedback:
        raise NotImplementedError


class RAGEvoAgent(EvoAgent[RAGEvaluator]):

    def __init__(
        self,
        max_loop: int,
        evolving_strategy: EvolvingStrategy,
        rag: Any,
        *,
        with_knowledge: bool = False,
        with_feedback: bool = True,
        knowledge_self_gen: bool = False,
    ) -> None:
        super().__init__(max_loop, evolving_strategy)
        self.rag = rag
        self.evolving_trace: list[EvoStep] = []
        self.with_knowledge = with_knowledge
        self.with_feedback = with_feedback
        self.knowledge_self_gen = knowledge_self_gen

    def multistep_evolve(
        self,
        evo: EvolvableSubjects,
        eva: RAGEvaluator | Feedback,
    ) -> Generator[EvolvableSubjects, None, None]:
        for evo_loop_id in tqdm(range(self.max_loop), "Implementing"):
            with logger.tag(f"evo_loop_{evo_loop_id}"):
                # 1. knowledge self-evolving
                if self.knowledge_self_gen and self.rag is not None:
                    self.rag.generate_knowledge(self.evolving_trace)
                # 2. RAG
                queried_knowledge = None
                if self.with_knowledge and self.rag is not None:
                    # TODO: Putting the evolving trace in here doesn't actually work
                    queried_knowledge = self.rag.query(evo, self.evolving_trace)

                # 3. evolve
                evo = self.evolving_strategy.evolve(
                    evo=evo,
                    evolving_trace=self.evolving_trace,
                    queried_knowledge=queried_knowledge,
                )

                # 4. Pack evolve results
                es = EvoStep(evo, queried_knowledge)

                # 5. Evaluation
                if self.with_feedback:
                    es.feedback = (
                        eva if isinstance(eva, Feedback) else eva.evaluate(evo, queried_knowledge=queried_knowledge)
                    )
                    logger.log_object(es.feedback, tag="evolving feedback")

                # 6. update trace
                self.evolving_trace.append(es)

                yield evo  # yield the control to caller for process control and logging.

                # 7. check if all tasks are completed
                if self.with_feedback and es.feedback is not None and es.feedback.finished():
                    logger.info("All tasks in evolving subject have been completed.")
                    break
