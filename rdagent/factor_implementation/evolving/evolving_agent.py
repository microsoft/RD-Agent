from rdagent.core.evolving_framework import Feedback, EvolvableSubjects, Evaluator, EvoStep

class EvoAgent:
    def __init__(self, evolving_strategy, rag) -> None:
        self.evolving_strategy = evolving_strategy
        self.rag = rag
        self.evolving_trace = []

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

        # 7. 更新trace
        self.evolving_trace.append(es)

        # Update trace to evo
        for index, feedback in enumerate(es.feedback):
            if feedback is not None:
                evo.evolve_trace[evo.target_factor_tasks[index].factor_name][-1].feedback = feedback
        
        return evo