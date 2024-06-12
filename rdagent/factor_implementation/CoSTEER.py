import pickle
from tqdm import tqdm
from pathlib import Path
from typing import List
from rdagent.core.implementation import TaskGenerator
from rdagent.core.task import TaskImplementation
from rdagent.factor_implementation.evolving.factor import FactorImplementTask, FactorEvovlingItem
from rdagent.knowledge_management.knowledgebase import FactorImplementationGraphKnowledgeBase, FactorImplementationGraphRAGStrategy
from rdagent.factor_implementation.evolving.evolving_strategy import FactorEvolvingStrategyWithGraph
from rdagent.factor_implementation.evolving.evaluators import FactorImplementationsMultiEvaluator, FactorImplementationEvaluatorV1
from rdagent.factor_implementation.evolving.evolving_agent import EvoAgent

class CoSTEERFG(TaskGenerator):
    def __init__(
        self,
        max_loops: int = 2,
        selection_method: str = "random",
        selection_ratio: float = 0.5,
        knowledge_base_path: Path = None,
        new_knowledge_base_path: Path = None,
        with_knowledge: bool = True,
        with_feedback: bool = True,
        knowledge_self_gen: bool = True,
    ) -> None:
        self.max_loops = max_loops
        self.selection_method = selection_method
        self.selection_ratio = selection_ratio
        self.knowledge_base_path = knowledge_base_path
        self.new_knowledge_base_path = new_knowledge_base_path
        self.with_knowledge = with_knowledge
        self.with_feedback = with_feedback
        self.knowledge_self_gen = knowledge_self_gen
        if self.knowledge_base_path is not None:
            self.knowledge_base_path = Path(knowledge_base_path)       # declare the evolving strategy and RAG strategy
        self.evolving_strategy = FactorEvolvingStrategyWithGraph()
        # declare the factor evaluator
        self.factor_evaluator = FactorImplementationsMultiEvaluator(FactorImplementationEvaluatorV1())

    def load_or_init_knowledge_base(self, former_knowledge_base_path: Path = None, component_init_list: list = []):
        if former_knowledge_base_path is not None and former_knowledge_base_path.exists():
            factor_knowledge_base = pickle.load(open(former_knowledge_base_path, "rb"))
            if not isinstance(
                factor_knowledge_base,
                FactorImplementationGraphKnowledgeBase,
            ):
                raise ValueError("The former knowledge base is not compatible with the current version")
        else:
            factor_knowledge_base = (
                FactorImplementationGraphKnowledgeBase(
                    init_component_list=component_init_list,
                )
            )
        return factor_knowledge_base
    
    def generate(self, tasks: List[FactorImplementTask]) -> List[TaskImplementation]:
        # init knowledge base
        factor_knowledge_base = self.load_or_init_knowledge_base(
            former_knowledge_base_path=self.knowledge_base_path,
            component_init_list=[],
        )
        # init rag method
        self.rag = (
            FactorImplementationGraphRAGStrategy(factor_knowledge_base)
        )

        # init indermediate items
        factor_implementations = FactorEvovlingItem(target_factor_tasks=tasks)

        self.evolve_agent = EvoAgent(evolving_strategy=self.evolving_strategy, rag=self.rag)

        for _ in tqdm(range(self.max_loops), "Implementing factors"):
            factor_implementations = self.evolve_agent.step_evolving(
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