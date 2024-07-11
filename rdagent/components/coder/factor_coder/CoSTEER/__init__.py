import pickle
from pathlib import Path

from rdagent.components.coder.factor_coder.config import FACTOR_IMPLEMENT_SETTINGS
from rdagent.components.coder.factor_coder.CoSTEER.evaluators import (
    FactorEvaluatorForCoder,
    FactorMultiEvaluator,
)
from rdagent.components.coder.factor_coder.CoSTEER.evolvable_subjects import (
    FactorEvolvingItem,
)
from rdagent.components.coder.factor_coder.CoSTEER.evolving_strategy import (
    FactorEvolvingStrategyWithGraph,
)
from rdagent.components.coder.factor_coder.CoSTEER.knowledge_management import (
    FactorGraphKnowledgeBase,
    FactorGraphRAGStrategy,
    FactorKnowledgeBaseV1,
)
from rdagent.components.coder.factor_coder.factor import FactorExperiment
from rdagent.core.evolving_agent import RAGEvoAgent
from rdagent.core.scenario import Scenario
from rdagent.core.task_generator import TaskGenerator


class FactorCoSTEER(TaskGenerator[FactorExperiment]):
    def __init__(
        self,
        *args,
        with_knowledge: bool = True,
        with_feedback: bool = True,
        knowledge_self_gen: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_loop = FACTOR_IMPLEMENT_SETTINGS.max_loop
        self.knowledge_base_path = (
            Path(FACTOR_IMPLEMENT_SETTINGS.knowledge_base_path)
            if FACTOR_IMPLEMENT_SETTINGS.knowledge_base_path is not None
            else None
        )
        self.new_knowledge_base_path = (
            Path(FACTOR_IMPLEMENT_SETTINGS.new_knowledge_base_path)
            if FACTOR_IMPLEMENT_SETTINGS.new_knowledge_base_path is not None
            else None
        )
        self.with_knowledge = with_knowledge
        self.with_feedback = with_feedback
        self.knowledge_self_gen = knowledge_self_gen
        self.evolving_strategy = FactorEvolvingStrategyWithGraph(scen=self.scen)
        # declare the factor evaluator
        self.factor_evaluator = FactorMultiEvaluator(FactorEvaluatorForCoder(scen=self.scen), scen=self.scen)
        self.evolving_version = 2

    def load_or_init_knowledge_base(self, former_knowledge_base_path: Path = None, component_init_list: list = []):
        if former_knowledge_base_path is not None and former_knowledge_base_path.exists():
            factor_knowledge_base = pickle.load(open(former_knowledge_base_path, "rb"))
            if self.evolving_version == 1 and not isinstance(factor_knowledge_base, FactorKnowledgeBaseV1):
                raise ValueError("The former knowledge base is not compatible with the current version")
            elif self.evolving_version == 2 and not isinstance(
                factor_knowledge_base,
                FactorGraphKnowledgeBase,
            ):
                raise ValueError("The former knowledge base is not compatible with the current version")
        else:
            factor_knowledge_base = (
                FactorGraphKnowledgeBase(
                    init_component_list=component_init_list,
                )
                if self.evolving_version == 2
                else FactorKnowledgeBaseV1()
            )
        return factor_knowledge_base

    def generate(self, exp: FactorExperiment) -> FactorExperiment:
        # init knowledge base
        factor_knowledge_base = self.load_or_init_knowledge_base(
            former_knowledge_base_path=self.knowledge_base_path,
            component_init_list=[],
        )
        # init rag method
        self.rag = FactorGraphRAGStrategy(factor_knowledge_base)

        # init intermediate items
        factor_experiment = FactorEvolvingItem(sub_tasks=exp.sub_tasks)

        self.evolve_agent = RAGEvoAgent(max_loop=self.max_loop, evolving_strategy=self.evolving_strategy, rag=self.rag)

        factor_experiment = self.evolve_agent.multistep_evolve(
            factor_experiment,
            self.factor_evaluator,
            with_knowledge=self.with_knowledge,
            with_feedback=self.with_feedback,
            knowledge_self_gen=self.knowledge_self_gen,
        )

        # save new knowledge base
        if self.new_knowledge_base_path is not None:
            pickle.dump(factor_knowledge_base, open(self.new_knowledge_base_path, "wb"))
        self.knowledge_base = factor_knowledge_base
        factor_experiment.based_experiments = exp.based_experiments
        return factor_experiment
