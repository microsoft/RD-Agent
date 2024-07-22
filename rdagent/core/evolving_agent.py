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
        **kwargs: Any,
    ) -> EvolvableSubjects: ...

    @abstractmethod
    def filter_evolvable_subjects_by_feedback(
        self,
        evo: EvolvableSubjects,
        feedback: Feedback | None,
    ) -> EvolvableSubjects: ...


class RAGEvoAgent(EvoAgent):
    def __init__(self, max_loop: int, evolving_strategy: EvolvingStrategy, rag: Any) -> None:
        super().__init__(max_loop, evolving_strategy)
        self.rag = rag
        self.evolving_trace: list[EvoStep] = []

    def multistep_evolve(
        self,
        evo: EvolvableSubjects,
        eva: Evaluator | Feedback,
        **kwargs: Any,
    ) -> EvolvableSubjects:
        with_knowledge = kwargs.get("with_knowledge", False)
        with_feedback = kwargs.get("with_feedback", True)
        knowledge_self_gen = kwargs.get("knowledge_self_gen", False)
        filter_final_evo = kwargs.get("filter_final_evo", False)

        for _ in tqdm(range(self.max_loop), "Implementing"):
            # 1. knowledge self-evolving
            if knowledge_self_gen and self.rag is not None:
                self.rag.generate_knowledge(self.evolving_trace)
            # 2. RAG
            queried_knowledge = None
            if with_knowledge and self.rag is not None:
                # TODO: Putting the evolving trace in here doesn't actually work
                queried_knowledge = self.rag.query(evo, self.evolving_trace)

            # 3. evolve
            evo = self.evolving_strategy.evolve(
                evo=evo,
                evolving_trace=self.evolving_trace,
                queried_knowledge=queried_knowledge,
            )
            logger.log_object(evo.sub_workspace_list, tag="evolving code")

            # 4. Pack evolve results
            es = EvoStep(evo, queried_knowledge)

            # 5. Evaluation
            if with_feedback:
                es.feedback = (
                    # TODO: Due to the irregular design of rdagent.core.evaluation.Evaluator,
                    # it fails mypy's test here, so we'll ignore this error for now.
                    eva if isinstance(eva, Feedback) else eva.evaluate(evo, queried_knowledge=queried_knowledge)  # type: ignore[arg-type, call-arg]
                )
                logger.log_object(es.feedback, tag="evolving feedback")

            # 6. update trace
            self.evolving_trace.append(es)
        if with_feedback and filter_final_evo:
            evo = self.filter_evolvable_subjects_by_feedback(evo, self.evolving_trace[-1].feedback)
        return evo
