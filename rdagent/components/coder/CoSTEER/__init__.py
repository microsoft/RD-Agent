import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERRAGStrategyV1,
    CoSTEERRAGStrategyV2,
)
from rdagent.core.developer import Developer
from rdagent.core.evolving_agent import EvolvingStrategy, RAGEvaluator, RAGEvoAgent
from rdagent.core.exception import CoderError
from rdagent.core.experiment import Experiment
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import RD_Agent_TIMER_wrapper


class CoSTEER(Developer[Experiment]):
    def __init__(
        self,
        settings: CoSTEERSettings,
        eva: RAGEvaluator,
        es: EvolvingStrategy,
        *args,
        evolving_version: int = 2,
        max_seconds: int | None = None,
        with_knowledge: bool = True,
        knowledge_self_gen: bool = True,
        max_loop: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.settings = settings

        self.max_loop = settings.max_loop if max_loop is None else max_loop
        self.max_seconds = max_seconds
        self.knowledge_base_path = (
            Path(settings.knowledge_base_path) if settings.knowledge_base_path is not None else None
        )
        self.new_knowledge_base_path = (
            Path(settings.new_knowledge_base_path) if settings.new_knowledge_base_path is not None else None
        )

        self.with_knowledge = with_knowledge
        self.knowledge_self_gen = knowledge_self_gen
        self.evolving_strategy = es
        self.evaluator = eva
        self.evolving_version = evolving_version

        # init rag method
        self.rag = (
            CoSTEERRAGStrategyV2(
                settings=settings,
                former_knowledge_base_path=self.knowledge_base_path,
                dump_knowledge_base_path=self.new_knowledge_base_path,
                evolving_version=self.evolving_version,
            )
            if self.evolving_version == 2
            else CoSTEERRAGStrategyV1(
                settings=settings,
                former_knowledge_base_path=self.knowledge_base_path,
                dump_knowledge_base_path=self.new_knowledge_base_path,
                evolving_version=self.evolving_version,
            )
        )

    def _get_last_fb(self) -> CoSTEERMultiFeedback:
        fb = self.evolve_agent.evolving_trace[-1].feedback
        assert fb is not None, "feedback is None"
        assert isinstance(fb, CoSTEERMultiFeedback), "feedback must be of type CoSTEERMultiFeedback"
        return fb

    def develop(self, exp: Experiment) -> Experiment:

        # init intermediate items
        evo_exp = EvolvingItem.from_experiment(exp)

        self.evolve_agent = RAGEvoAgent[EvolvingItem](
            max_loop=self.max_loop,
            evolving_strategy=self.evolving_strategy,
            rag=self.rag,
            with_knowledge=self.with_knowledge,
            with_feedback=True,
            knowledge_self_gen=self.knowledge_self_gen,
            enable_filelock=self.settings.enable_filelock,
            filelock_path=self.settings.filelock_path,
        )

        # Evolving the solution
        start_datetime = datetime.now()
        fallback_evo_exp = None
        for evo_exp in self.evolve_agent.multistep_evolve(evo_exp, self.evaluator):
            assert isinstance(evo_exp, Experiment)  # multiple inheritance
            if self._get_last_fb().is_acceptable():
                fallback_evo_exp = deepcopy(evo_exp)
                fallback_evo_exp.create_ws_ckp()  # NOTE: creating checkpoints for saving files in the workspace to prevent inplace mutation.

            logger.log_object(evo_exp.sub_workspace_list, tag="evolving code")
            for sw in evo_exp.sub_workspace_list:
                logger.info(f"evolving workspace: {sw}")
            if self.max_seconds is not None and (datetime.now() - start_datetime).seconds > self.max_seconds:
                logger.info(f"Reached max time limit {self.max_seconds} seconds, stop evolving")
                break
            if RD_Agent_TIMER_wrapper.timer.started and RD_Agent_TIMER_wrapper.timer.is_timeout():
                logger.info("Global timer is timeout, stop evolving")
                break

        # if the final feedback is not finished(therefore acceptable), we will use the fallback solution.
        try:
            evo_exp = self._exp_postprocess_by_feedback(evo_exp, self._get_last_fb())
        except CoderError:
            if fallback_evo_exp is not None:
                logger.info("Fallback to the fallback solution.")
                evo_exp = fallback_evo_exp
                evo_exp.recover_ws_ckp()  # NOTE: recovering checkpoints for restoring files in the workspace to prevent inplace mutation.
            else:
                raise

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
