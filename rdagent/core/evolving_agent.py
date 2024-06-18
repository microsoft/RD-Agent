from rdagent.core.evaluation import Evaluator
from rdagent.core.evolving_framework import Feedback, EvolvableSubjects, EvoStep
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Any


class EvoAgent(ABC):
    def __init__(self, max_loop, evolving_strategy) -> None:
        self.max_loop = max_loop
        self.evolving_strategy = evolving_strategy

    @abstractmethod
    def multistep_evolve(self, evo: EvolvableSubjects, eva: Evaluator | Feedback, **kwargs: Any) -> EvolvableSubjects:
        pass


class RAGEvoAgent(EvoAgent):
    def __init__(self, max_loop, evolving_strategy, rag) -> None:
        super().__init__(max_loop, evolving_strategy)
        self.rag = rag
        self.evolving_trace = []

    def multistep_evolve(
        self,
        evo: EvolvableSubjects,
        eva: Evaluator | Feedback,
        *,
        with_knowledge: bool = False,
        with_feedback: bool = True,
        knowledge_self_gen: bool = False,
    ) -> EvolvableSubjects:

        for _ in tqdm(range(self.max_loop), "Implementing factors"):
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

            # 4. Pack evolve results
            es = EvoStep(evo, queried_knowledge)

            # 5. Evaluation
            if with_feedback:
                es.feedback = (
                    eva if isinstance(eva, Feedback) else eva.evaluate(evo, queried_knowledge=queried_knowledge)
                )

            # 6. update trace
            self.evolving_trace.append(es)

        return evo
