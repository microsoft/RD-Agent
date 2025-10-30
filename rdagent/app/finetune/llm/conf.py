import os
from pathlib import Path

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import RD_AGENT_SETTINGS, ExtendedBaseSettings


class LLMFinetunePropSetting(ExtendedBaseSettings):
    """LLM Fine-tune dedicated property settings.

    - Adjust timeouts and template
    - Use FT_ env prefix for overrides
    """

    model_config = SettingsConfigDict(env_prefix="FT_", protected_namespaces=())

    # Main Components
    scen: str = "rdagent.scenarios.finetune.scen.scenario.LLMFinetuneScen"
    """Scenario class for LLM fine-tuning tasks."""

    hypothesis_gen: str = "rdagent.scenarios.finetune.proposal.proposal.LLMFinetuneExpGen"
    """Hypothesis generation class for LLM fine-tuning tasks."""

    hypothesis2experiment: str = "rdagent.scenarios.finetune.proposal.proposal.FTHypothesis2Experiment"
    """Hypothesis to experiment converter.
    Function: Convert abstract LLM fine-tuning hypotheses into concrete experiment configurations.
    """

    coder: str = "rdagent.components.coder.finetune.LLMFinetuneCoSTEER"
    """Code generator.
    Function: Generate LLM fine-tuning code based on experiment design.
    """

    runner: str = "rdagent.scenarios.finetune.train.runner.LLMFinetuneRunner"  # TODO
    """Code runner.
    Function: Execute LLM fine-tuning code in a Docker environment.
    """

    summarizer: str = "rdagent.scenarios.finetune.dev.feedback.FTExperiment2Feedback"
    """Result summarizer - To be implemented.
    Function: Analyze fine-tuning results and generate feedback, including performance metrics and error analysis.
    """

    # Timeouts (longer for LLM training)
    debug_timeout: int = 36000
    debug_recommend_timeout: int = 36000
    full_timeout: int = 360000
    full_recommend_timeout: int = 360000

    # Pipeline behavior
    coder_on_whole_pipeline: bool = True
    app_tpl: str = "scenarios/finetune"

    # Benchmark evaluation (always enabled as part of evaluation pipeline)
    benchmark_datasets: list[str] = ["gsm8k"]
    """Benchmark datasets to evaluate on. Supported: aime25, mmlu, gsm8k, humaneval, bbh, hellaswag, cmmlu, arc, etc."""

    benchmark_timeout: int = 3600
    """Benchmark evaluation timeout in seconds"""

    benchmark_limit: int | None = None
    """Limit number of samples for benchmark evaluation (None for full evaluation). Use for quick testing and debugging."""

    # Data paths and processing
    file_path: str | None = None  # FT_FILE_PATH/datasets/<dataset>/, FT_FILE_PATH/models/<baseModel>/
    show_nan_columns: bool = False
    sample_data_by_LLM: bool = True

    # LLM-specific fields
    base_model: str | None = None
    dataset: str = ""

    # LLaMA Factory
    update_llama_factory: bool = True

    # Docker settings
    docker_enable_cache: bool = False
    """Enable Docker cache for training (set via FT_DOCKER_ENABLE_CACHE)"""

    @property
    def task(self) -> str:
        """Generate task name from base model and dataset."""
        if self.base_model and self.dataset:
            return f"{self.base_model}@{self.dataset}".replace("/", "_").replace("\\", "_")
        return ""


# Global setting instance for LLM finetuning scenario
FT_RD_SETTING = LLMFinetunePropSetting()
