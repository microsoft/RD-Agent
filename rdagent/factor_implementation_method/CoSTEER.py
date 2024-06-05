import pickle
from pathlib import Path
from typing import List
from rdagent.core.implementation import TaskGenerator
from rdagent.core.task import FactorTask, TaskImplementation
from rdagent.knowledge_management.knowledgebase import FactorImplementationGraphKnowledgeBase, FactorImplementationGraphRAGStrategy
from rdagent.factor_implementation_method.evolving_strategy import FactorEvolvingStrategyWithGraph, FactorImplementationGraphRAGStrategy
from tqdm import tqdm

class CoSTEERFG(TaskGenerator):
    def __init__(
        self,
        max_loops: int = 10,
        selection_method: str = "random",
        selection_ratio: float = 0.5,
        knowledge_base_path: Path = None,
        new_knowledge_base_path: Path = None,
        with_knowledge: bool = True,
        with_feedback: bool = True,
    ) -> None:
        self.max_loops = max_loops
        self.selection_method = selection_method
        self.selection_ratio = selection_ratio
        self.knowledge_base_path = knowledge_base_path
        self.new_knowledge_base_path = new_knowledge_base_path
        self.with_knowledge = with_knowledge
        self.with_feedback = with_feedback
        if self.knowledge_base_path is not None:
            self.knowledge_base        # declare the evolving strategy and RAG strategy
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
    
    def generate(self, tasks: List[FactorTask]) -> List[TaskImplementation]:
        # init knowledge base
        factor_knowledge_base = self.load_or_init_knowledge_base(
            former_knowledge_base_path=self.knowledge_base,
            component_init_list=[],
        )
        # rag变量里面还有prompt？
        self.rag = (
            FactorImplementationGraphRAGStrategy(factor_knowledge_base)
        )        
        for _ in tqdm(range(self.max_loops), "Implementing factors"):
            factor_implementations = self.step_evolving(
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

    def step_evolving(
        self,
        evo: EvolvableSubjects,
        eva: Evaluator | Feedback,
        *,
        with_knowledge: bool = False,
        with_feedback: bool = True,
        knowledge_self_gen: bool = False,
    ) -> EvolvableSubjects:
        # 1. knowledge self-evolving
        if knowledge_self_gen and self.rag is not None:
            self.rag.generate_knowledge(self.evolving_trace)

        # 2. 检索需要的Knowledge
        queried_knowledge = None
        if with_knowledge and self.rag is not None:
            # 这里放了evolving_trace实际上没有作用
            queried_knowledge = self.rag.query(evo, self.evolving_trace)

        # 3. evolve
        evo = self.evolving_strategy.evolve(
            evo=evo,
            evolving_trace=self.evolving_trace,
            queried_knowledge=queried_knowledge,
        )
        # 4. 封装Evolve结果
        es = EvoStep(evo, queried_knowledge)

        # 5. 环境评测反馈
        if with_feedback:
            es.feedback = eva if isinstance(eva, Feedback) else eva.evaluate(evo, queried_knowledge=queried_knowledge)

        # 6. 单轮ic值计算
        ev = FactorImplementationCorrelationEvaluator(hard_check=False)
        for index, imp in enumerate(evo.corresponding_implementations):
            if imp is not None:
                try:
                    score = ev.evaluate(evo.corresponding_gt[index].ground_truth, imp)
                except:
                    score = 0
                evo.evolve_trace[evo.target_factor_tasks[index].factor_name][-1].score = score

        # 7. 更新trace
        self.evolving_trace.append(es)

        # Update trace to evo
        for index, feedback in enumerate(es.feedback):
            if feedback is not None:
                evo.evolve_trace[evo.target_factor_tasks[index].factor_name][-1].feedback = feedback
        
        return evo
