import pickle
from pathlib import Path

from rdagent.components.coder.model_coder.conf import MODEL_IMPL_SETTINGS
from rdagent.components.coder.model_coder.CoSTEER.evaluators import (
    ModelCoderMultiEvaluator,
)
from rdagent.components.coder.model_coder.CoSTEER.evolvable_subjects import (
    ModelEvolvingItem,
)
from rdagent.components.coder.model_coder.CoSTEER.evolving_agent import ModelRAGEvoAgent
from rdagent.components.coder.model_coder.CoSTEER.evolving_strategy import (
    ModelCoderEvolvingStrategy,
)
from rdagent.components.coder.model_coder.CoSTEER.knowledge_management import (
    ModelKnowledgeBase,
    ModelRAGStrategy,
)
from rdagent.components.coder.model_coder.model import ModelExperiment
from rdagent.core.developer import Developer
from rdagent.core.evolving_agent import RAGEvoAgent


class ModelCoSTEER(Developer[ModelExperiment]):
    def __init__(
        self,
        *args,
        with_knowledge: bool = True,
        with_feedback: bool = True,
        knowledge_self_gen: bool = True,
        filter_final_evo: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_loop = MODEL_IMPL_SETTINGS.max_loop
        self.knowledge_base_path = (
            Path(MODEL_IMPL_SETTINGS.knowledge_base_path)
            if MODEL_IMPL_SETTINGS.knowledge_base_path is not None
            else None
        )
        self.new_knowledge_base_path = (
            Path(MODEL_IMPL_SETTINGS.new_knowledge_base_path)
            if MODEL_IMPL_SETTINGS.new_knowledge_base_path is not None
            else None
        )
        self.with_knowledge = with_knowledge
        self.with_feedback = with_feedback
        self.knowledge_self_gen = knowledge_self_gen
        self.filter_final_evo = filter_final_evo
        self.evolving_strategy = ModelCoderEvolvingStrategy(scen=self.scen)
        self.model_evaluator = ModelCoderMultiEvaluator(scen=self.scen)

    def load_or_init_knowledge_base(self, former_knowledge_base_path: Path = None, component_init_list: list = []):
        if former_knowledge_base_path is not None and former_knowledge_base_path.exists():
            model_knowledge_base = pickle.load(open(former_knowledge_base_path, "rb"))
            if not isinstance(model_knowledge_base, ModelKnowledgeBase):
                raise ValueError("The former knowledge base is not compatible with the current version")
        else:
            model_knowledge_base = ModelKnowledgeBase()

        return model_knowledge_base

    def develop(self, exp: ModelExperiment) -> ModelExperiment:
        # init knowledge base
        model_knowledge_base = self.load_or_init_knowledge_base(
            former_knowledge_base_path=self.knowledge_base_path,
            component_init_list=[],
        )
        # init rag method
        self.rag = ModelRAGStrategy(model_knowledge_base)

        # init intermediate items
        model_experiment = ModelEvolvingItem(sub_tasks=exp.sub_tasks)

        self.evolve_agent = ModelRAGEvoAgent(
            max_loop=self.max_loop, evolving_strategy=self.evolving_strategy, rag=self.rag
        )

        model_experiment = self.evolve_agent.multistep_evolve(
            model_experiment,
            self.model_evaluator,
            with_knowledge=self.with_knowledge,
            with_feedback=self.with_feedback,
            knowledge_self_gen=self.knowledge_self_gen,
            filter_final_evo=self.filter_final_evo,
        )

        # save new knowledge base
        if self.new_knowledge_base_path is not None:
            pickle.dump(model_knowledge_base, open(self.new_knowledge_base_path, "wb"))
        exp.sub_workspace_list = model_experiment.sub_workspace_list
        return exp
