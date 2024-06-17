import pickle
from pathlib import Path
from typing import List
from rdagent.core.implementation import TaskGenerator
from rdagent.core.task import TaskImplementation
from rdagent.factor_implementation.evolving.knowledge_management import FactorImplementationKnowledgeBaseV1
from rdagent.factor_implementation.evolving.factor import FactorImplementTask, FactorEvovlingItem
from rdagent.knowledge_management.knowledgebase import (
    FactorImplementationGraphKnowledgeBase,
    FactorImplementationGraphRAGStrategy,
)
from rdagent.factor_implementation.evolving.evolving_strategy import FactorEvolvingStrategyWithGraph
from rdagent.factor_implementation.evolving.evaluators import (
    FactorImplementationsMultiEvaluator,
    FactorImplementationEvaluatorV1,
)
from rdagent.core.evolving_agent import RAGEvoAgent
from rdagent.factor_implementation.share_modules.factor_implementation_config import (
    Factor_Implement_Settings,
)


class CoSTEERFG(TaskGenerator):
    def __init__(
        self,
        with_knowledge: bool = True,
        with_feedback: bool = True,
        knowledge_self_gen: bool = True,
    ) -> None:
        self.max_loop = Factor_Implement_Settings.max_loop
        self.knowledge_base_path = (
            Path(Factor_Implement_Settings.knowledge_base_path)
            if Factor_Implement_Settings.knowledge_base_path is not None
            else None
        )
        self.new_knowledge_base_path = (
            Path(Factor_Implement_Settings.new_knowledge_base_path)
            if Factor_Implement_Settings.new_knowledge_base_path is not None
            else None
        )
        self.with_knowledge = with_knowledge
        self.with_feedback = with_feedback
        self.knowledge_self_gen = knowledge_self_gen
        self.evolving_strategy = FactorEvolvingStrategyWithGraph()
        # declare the factor evaluator
        self.factor_evaluator = FactorImplementationsMultiEvaluator(FactorImplementationEvaluatorV1())
        self.evolving_version = 2

    def load_or_init_knowledge_base(self, former_knowledge_base_path: Path = None, component_init_list: list = []):

        if former_knowledge_base_path is not None and former_knowledge_base_path.exists():
            factor_knowledge_base = pickle.load(open(former_knowledge_base_path, "rb"))
            if self.evolving_version == 1 and not isinstance(
                factor_knowledge_base, FactorImplementationKnowledgeBaseV1
            ):
                raise ValueError("The former knowledge base is not compatible with the current version")
            elif self.evolving_version == 2 and not isinstance(
                factor_knowledge_base,
                FactorImplementationGraphKnowledgeBase,
            ):
                raise ValueError("The former knowledge base is not compatible with the current version")
        else:
            factor_knowledge_base = (
                FactorImplementationGraphKnowledgeBase(
                    init_component_list=component_init_list,
                )
                if self.evolving_version == 2
                else FactorImplementationKnowledgeBaseV1()
            )
        return factor_knowledge_base

    def generate(self, tasks: List[FactorImplementTask]) -> List[TaskImplementation]:
        # init knowledge base
        factor_knowledge_base = self.load_or_init_knowledge_base(
            former_knowledge_base_path=self.knowledge_base_path,
            component_init_list=[],
        )
        # init rag method
        self.rag = FactorImplementationGraphRAGStrategy(factor_knowledge_base)

        # init indermediate items
        factor_implementations = FactorEvovlingItem(target_factor_tasks=tasks)

        self.evolve_agent = RAGEvoAgent(max_loop=self.max_loop, evolving_strategy=self.evolving_strategy, rag=self.rag)

        factor_implementations = self.evolve_agent.multistep_evolve(
            factor_implementations,
            self.factor_evaluator,
            with_knowledge=self.with_knowledge,
            with_feedback=self.with_feedback,
            knowledge_self_gen=self.knowledge_self_gen,
        )

        # save new knowledge base
        if self.new_knowledge_base_path is not None:
            pickle.dump(factor_knowledge_base, open(self.new_knowledge_base_path, "wb"))
        self.knowledge_base = factor_knowledge_base
        self.latest_factor_implementations = tasks
        return factor_implementations
