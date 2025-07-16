import pickle
from datetime import datetime
from pathlib import Path

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERKnowledgeBaseV1,
    CoSTEERKnowledgeBaseV2,
    CoSTEERRAGStrategyV1,
    CoSTEERRAGStrategyV2,
)
from rdagent.core.developer import Developer
from rdagent.core.evaluation import Evaluator, Feedback
from rdagent.core.evolving_agent import EvolvingStrategy, RAGEvoAgent
from rdagent.core.exception import CoderError
from rdagent.core.experiment import Experiment
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import RD_Agent_TIMER_wrapper


class CoSTEER(Developer[Experiment]):
    def __init__(
        self,
        settings: CoSTEERSettings,
        eva: Evaluator,
        es: EvolvingStrategy,
        evolving_version: int,
        *args,
        with_knowledge: bool = True,
        with_feedback: bool = True,
        knowledge_self_gen: bool = True,
        filter_final_evo: bool = True,
        max_loop: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_loop = settings.max_loop if max_loop is None else max_loop
        self.max_seconds = settings.max_seconds
        self.knowledge_base_path = (
            Path(settings.knowledge_base_path) if settings.knowledge_base_path is not None else None
        )
        self.new_knowledge_base_path = (
            Path(settings.new_knowledge_base_path) if settings.new_knowledge_base_path is not None else None
        )

        self.with_knowledge = with_knowledge
        self.with_feedback = with_feedback
        self.knowledge_self_gen = knowledge_self_gen
        self.filter_final_evo = filter_final_evo
        self.evolving_strategy = es
        self.evaluator = eva
        self.evolving_version = evolving_version

        # init knowledge base
        self.knowledge_base = self.load_or_init_knowledge_base(
            former_knowledge_base_path=self.knowledge_base_path,
            component_init_list=[],
        )
        # init rag method
        self.rag = (
            CoSTEERRAGStrategyV2(self.knowledge_base, settings=settings)
            if self.evolving_version == 2
            else CoSTEERRAGStrategyV1(self.knowledge_base, settings=settings)
        )

    def load_or_init_knowledge_base(self, former_knowledge_base_path: Path = None, component_init_list: list = []):
        if former_knowledge_base_path is not None and former_knowledge_base_path.exists():
            knowledge_base = pickle.load(open(former_knowledge_base_path, "rb"))
            if self.evolving_version == 1 and not isinstance(knowledge_base, CoSTEERKnowledgeBaseV1):
                raise ValueError("The former knowledge base is not compatible with the current version")
            elif self.evolving_version == 2 and not isinstance(
                knowledge_base,
                CoSTEERKnowledgeBaseV2,
            ):
                raise ValueError("The former knowledge base is not compatible with the current version")
        else:
            knowledge_base = (
                CoSTEERKnowledgeBaseV2(
                    init_component_list=component_init_list,
                )
                if self.evolving_version == 2
                else CoSTEERKnowledgeBaseV1()
            )
        return knowledge_base

    def develop(self, exp: Experiment) -> Experiment:

        # init intermediate items
        evo_exp = EvolvingItem.from_experiment(exp)

        self.evolve_agent = RAGEvoAgent(
            max_loop=self.max_loop,
            evolving_strategy=self.evolving_strategy,
            rag=self.rag,
            with_knowledge=self.with_knowledge,
            with_feedback=self.with_feedback,
            knowledge_self_gen=self.knowledge_self_gen,
        )

        start_datetime = datetime.now()
        for evo_exp in self.evolve_agent.multistep_evolve(evo_exp, self.evaluator):
            assert isinstance(evo_exp, Experiment)  # multiple inheritance
            logger.log_object(evo_exp.sub_workspace_list, tag="evolving code")
            for sw in evo_exp.sub_workspace_list:
                logger.info(f"evolving workspace: {sw}")
            if (datetime.now() - start_datetime).seconds > self.max_seconds:
                logger.info(f"Reached max time limit {self.max_seconds} seconds, stop evolving")
                break
            if RD_Agent_TIMER_wrapper.timer.started and RD_Agent_TIMER_wrapper.timer.is_timeout():
                logger.info("Global timer is timeout, stop evolving")
                break

        if self.with_feedback and self.filter_final_evo:
            evo_exp = self._exp_postprocess_by_feedback(evo_exp, self.evolve_agent.evolving_trace[-1].feedback)

        # save new knowledge base
        if self.new_knowledge_base_path is not None:
            with self.new_knowledge_base_path.open("wb") as f:
                pickle.dump(self.knowledge_base, f)
            logger.info(f"New knowledge base saved to {self.new_knowledge_base_path}")
        exp.sub_workspace_list = evo_exp.sub_workspace_list
        exp.experiment_workspace = evo_exp.experiment_workspace
        return exp

    def _exp_postprocess_by_feedback(self, evo: Experiment, feedback: CoSTEERMultiFeedback) -> Experiment:
        """
        Responsibility:
        - Raise Error if it failed to handle the develop task
        -
        """
        assert isinstance(evo, Experiment)
        assert isinstance(feedback, CoSTEERMultiFeedback)
        assert len(evo.sub_workspace_list) == len(feedback)

        # FIXME: when whould the feedback be None?
        failed_feedbacks = [
            f"- feedback{index + 1:02d}:\n  - execution: {f.execution}\n  - return_checking: {f.return_checking}\n  - code: {f.code}"
            for index, f in enumerate(feedback)
            if f is not None and not f.final_decision
        ]

        if len(failed_feedbacks) == len(feedback):
            feedback_summary = "\n".join(failed_feedbacks)
            raise CoderError(f"All tasks are failed:\n{feedback_summary}")

        return evo
