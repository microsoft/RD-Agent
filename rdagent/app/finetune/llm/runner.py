from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.app.finetune.llm.eval import LLMFinetuneEvaluator, LLMPipelineEvaluator
from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.data_science.conf import DSCoderCoSTEERSettings
from rdagent.components.coder.data_science.share.eval import ModelDumpEvaluator
from rdagent.core.scenario import Scenario
from rdagent.scenarios.data_science.dev.runner import (
    DSRunnerMultiProcessEvolvingStrategy,
)


class LLMFinetunePipelineCoSTEERSettings(DSCoderCoSTEERSettings):
    """LLM Fine-tuning specific CoSTEER settings."""

    class Config:
        env_prefix = "LLM_FT_Pipeline_CoSTEER_"


class LLMFinetuneRunner(CoSTEER):
    """LLM Fine-tuning specific runner that uses LLM Docker environment."""

    def __init__(
        self,
        scen: Scenario,
        use_pipeline_evaluator: bool = True,
        *args,
        **kwargs,
    ) -> None:
        # Choose evaluator based on task type
        if use_pipeline_evaluator:
            eval_l = [LLMPipelineEvaluator(scen=scen)]
        else:
            eval_l = [LLMFinetuneEvaluator(scen=scen)]

        # Add model dump evaluator if enabled
        if FT_RD_SETTING.enable_model_dump:
            eval_l.append(ModelDumpEvaluator(scen=scen, data_type="full"))

        eva = CoSTEERMultiEvaluator(single_evaluator=eval_l, scen=scen)

        settings = LLMFinetunePipelineCoSTEERSettings()
        es = DSRunnerMultiProcessEvolvingStrategy(scen=scen, settings=settings)

        # Initialize with LLM-specific configuration
        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            max_loop=getattr(FT_RD_SETTING, "runner_max_loop", 3),  # Default to 3 loops
            **kwargs,
        )

    def get_develop_max_seconds(self) -> int | None:
        """Get maximum seconds for development using FT settings."""
        return int(self.scen.real_full_timeout() * self.settings.max_seconds_multiplier)

    def compare_and_pick_fb(self, base_fb, new_fb) -> bool:
        """Compare feedback for LLM fine-tuning results."""
        if base_fb is None:
            return True

        base_fb = base_fb[0]
        new_fb = new_fb[0]

        def compare_scores(s1, s2) -> bool:
            if s2 is None:
                return False
            if s1 is None:
                return True
            return (s2 > s1) == self.scen.metric_direction

        return compare_scores(getattr(base_fb, "score", None), getattr(new_fb, "score", None))

    def develop(self, exp):
        """Develop experiment using LLM-specific environment."""
        bak_sub_tasks = exp.pending_tasks_list
        try:
            return super().develop(exp)
        finally:
            exp.pending_tasks_list = bak_sub_tasks
