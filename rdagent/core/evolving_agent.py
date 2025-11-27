from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import nullcontext
from typing import Any, Generic, TypeVar

from filelock import FileLock
from tqdm import tqdm

from rdagent.core.evaluation import EvaluableObj, Evaluator, Feedback
from rdagent.core.evolving_framework import (
    EvolvableSubjects,
    EvolvingStrategy,
    EvoStep,
    IterEvaluator,
    RAGStrategy,
)
from rdagent.log import rdagent_logger as logger

ASpecificEvaluator = TypeVar("ASpecificEvaluator", bound=Evaluator)
ASpecificEvolvableSubjects = TypeVar("ASpecificEvolvableSubjects", bound=EvolvableSubjects)


class EvoAgent(ABC, Generic[ASpecificEvaluator, ASpecificEvolvableSubjects]):

    def __init__(self, max_loop: int, evolving_strategy: EvolvingStrategy) -> None:
        self.max_loop = max_loop
        self.evolving_strategy = evolving_strategy

    @abstractmethod
    def multistep_evolve(
        self,
        evo: ASpecificEvolvableSubjects,
        eva: ASpecificEvaluator,
    ) -> Generator[ASpecificEvolvableSubjects, None, None]:
        """
        yield EvolvableSubjects for caller for easier process control and logging.
        """


class RAGEvaluator(IterEvaluator):

    @abstractmethod
    def evaluate_iter(self, queried_knowledge: object = None, evolving_trace: list[EvoStep] = []) -> Generator[Feedback, EvaluableObj | None, Feedback]:
        """

        1) It will yield a evaluation for each implement part and yield the feedback for that part.
        2) And finally, it will get the summarize all the feedback and return a overall feedback.

        Sending a None feedback will stop the evaluation chain and just return the overall feedback.

        A typical implementation of this method is:

        .. code-block:: python

            evo = yield Feedback()  # it will receive the evo first, so the first yield is for get the sent evo instead of generate useful feedback
            assert evo is not None
            for partial_eval_func in self.evaluate_func_iter():
                partial_fb = partial_eval_func(evo, queried_knowledge, evolving_trace)
                # return the partial feedback and receive the evolved solution for next iteration
                yield partial_fb

            final_fb = get_final_fb(...)
            return final_fb

        """


def get_return_value(gen: Generator[Any, Any, Feedback]) -> Feedback:
    """get the return value from a generator"""
    try:
        next(gen)
    except StopIteration as e:
        return e.value


class RAGEvoAgent(EvoAgent[RAGEvaluator, ASpecificEvolvableSubjects], Generic[ASpecificEvolvableSubjects]):

    def __init__(
        self,
        max_loop: int,
        evolving_strategy: EvolvingStrategy,
        rag: RAGStrategy,
        *,
        with_knowledge: bool = False,
        knowledge_self_gen: bool = False,
        enable_filelock: bool = False,
        filelock_path: str | None = None,
        stop_eval_chain_on_fail: bool = False,
    ) -> None:
        """
        Initialize a Retrieval-Augmented Generation (RAG) based evolutionary agent.

        Args:
            max_loop (int): Maximum number of evolution loops to execute.
            evolving_strategy (EvolvingStrategy): Strategy defining how the subjects evolve each step.
            rag (RAGStrategy): Retrieval-Augmented Generation strategy instance used for knowledge querying and/or creation.
            with_knowledge (bool, optional): If True, retrieves knowledge from RAG for each evolution step. Defaults to False.
            knowledge_self_gen (bool, optional): If True, enable RAG to load, generate, dump new knowledge from evolving trace. Defaults to False.
            enable_filelock (bool, optional): If True, enables file-based lock when accessing/modifying the RAG knowledge base. Defaults to False.
            filelock_path (str | None, optional): Path to the lock file when enable_filelock is True. Defaults to None.

        This class coordinates the multi-step evolution process with optional:
            - Knowledge retrieval before evolving.
            - Feedback collection after evolving.
            - Self-generation and persisting of knowledge base updates.

        Evolving trace is maintained across steps for adaptive strategies and knowledge generation.
        """
        super().__init__(max_loop, evolving_strategy)
        self.rag = rag
        self.evolving_trace: list[EvoStep[ASpecificEvolvableSubjects]] = []
        self.with_knowledge = with_knowledge
        self.knowledge_self_gen = knowledge_self_gen
        self.enable_filelock = enable_filelock
        self.filelock_path = filelock_path
        self.stop_eval_chain_on_fail = stop_eval_chain_on_fail

    def multistep_evolve(
        self,
        evo: ASpecificEvolvableSubjects,
        eva: RAGEvaluator,
    ) -> Generator[ASpecificEvolvableSubjects, None, None]:
        for evo_loop_id in tqdm(range(self.max_loop), "Implementing"):
            with logger.tag(f"evo_loop_{evo_loop_id}"):
                # 1. RAG
                queried_knowledge = None
                if self.with_knowledge and self.rag is not None:
                    # TODO: Putting the evolving trace in here doesn't actually work
                    queried_knowledge = self.rag.query(evo, self.evolving_trace)

                # 2. evolve:
                # A compelete solution of an evo can be break down into multiple evolving steps.
                # Each evolving step can be evaluated separately.
                # Assumptions: 
                # - if we want to stop on some point of the implementation, we must have a according evaluator (Otherwise, It is meaningless to stop)
                evo_iter = self.evolving_strategy.evolve_iter(
                    evo=evo,
                    evolving_trace=self.evolving_trace,
                    queried_knowledge=queried_knowledge,
                )
                eva_iter = eva.evaluate_iter(
                    evolving_trace=self.evolving_trace,
                    queried_knowledge=queried_knowledge,
                )
                next(eva_iter) # kick off the first iteration
                for evo in evo_iter:
                    step_feedback = eva_iter.send(evo)
                    if not step_feedback and self.stop_eval_chain_on_fail:
                        eva_iter.send(None) # sening the signal to skip the rest partial evaluation and return the overall feedback directly
                        break
                overall_feedback = get_return_value(eva_iter)

                # 3. Pack evolve results
                es = EvoStep[ASpecificEvolvableSubjects](evo, queried_knowledge, overall_feedback)

                # 4. Evaluation
                logger.log_object(es.feedback, tag="evolving feedback")

                # 5. update trace
                self.evolving_trace.append(es)

                # 6. knowledge self-evolving
                if self.knowledge_self_gen and self.rag is not None:
                    with FileLock(self.filelock_path) if self.enable_filelock else nullcontext():  # type: ignore[arg-type]
                        self.rag.load_dumped_knowledge_base()
                        self.rag.generate_knowledge(self.evolving_trace)
                        self.rag.dump_knowledge_base()

                yield evo  # yield the control to caller for process control and logging.

                # 7. check if all tasks are completed
                if es.feedback.finished():
                    logger.info("All tasks in evolving subject have been completed.")
                    break
