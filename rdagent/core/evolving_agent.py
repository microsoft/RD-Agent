from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

if TYPE_CHECKING:
    from rdagent.core.evaluation import Evaluator
    from rdagent.core.evolving_framework import EvolvableSubjects

from rdagent.core.evaluation import Feedback
from rdagent.core.evolving_framework import EvolvingStrategy, EvoStep
from rdagent.log import rdagent_logger as logger


class EvoAgent(ABC):
    def __init__(self, max_loop: int, evolving_strategy: EvolvingStrategy) -> None:
        self.max_loop = max_loop
        self.evolving_strategy = evolving_strategy

    @abstractmethod
    def multistep_evolve(
        self,
        evo: EvolvableSubjects,
        eva: Evaluator | Feedback,
        filter_final_evo: bool = False,
    ) -> EvolvableSubjects: ...

    @abstractmethod
    def filter_evolvable_subjects_by_feedback(
        self,
        evo: EvolvableSubjects,
        feedback: Feedback | None,
    ) -> EvolvableSubjects: ...


class RAGEvoAgent(EvoAgent):
    def __init__(
        self,
        max_loop: int,
        evolving_strategy: EvolvingStrategy,
        rag: Any,
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
        eva: Evaluator | Feedback,
        filter_final_evo: bool = False,
    ) -> EvolvableSubjects:
        for _ in tqdm(range(self.max_loop), "Implementing"):
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
            # TODO: Due to design issues, we have chosen to ignore this mypy error.
            logger.log_object(evo.sub_workspace_list, tag="evolving code")  # type: ignore[attr-defined]
            for sw in evo.sub_workspace_list:  # type: ignore[attr-defined]
                logger.info(f"evolving code workspace: {sw}")

            # 4. Pack evolve results
            es = EvoStep(evo, queried_knowledge)

            # 5. Evaluation
            if self.with_feedback:
                es.feedback = (
                    # TODO: Due to the irregular design of rdagent.core.evaluation.Evaluator,
                    # it fails mypy's test here, so we'll ignore this error for now.
                    eva
                    if isinstance(eva, Feedback)
                    else eva.evaluate(evo, queried_knowledge=queried_knowledge)  # type: ignore[arg-type, call-arg]
                )
                logger.log_object(es.feedback, tag="evolving feedback")

            # 6. update trace
            self.evolving_trace.append(es)
        if self.with_feedback and filter_final_evo:
            evo = self.filter_evolvable_subjects_by_feedback(evo, self.evolving_trace[-1].feedback)
        return evo
