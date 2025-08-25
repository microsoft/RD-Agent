import os

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import RD_AGENT_SETTINGS, ExtendedBaseSettings
from rdagent.scenarios.finetune.utils import prev_model_dirname


class LLMFinetunePropSetting(ExtendedBaseSettings):
    """LLM Fine-tune dedicated property settings.

    - Adjust timeouts and template
    - Use FT_ env prefix for overrides
    """

    model_config = SettingsConfigDict(env_prefix="FT_", protected_namespaces=())

    # Main Components
    scen: str = "rdagent.scenarios.finetune.scen.LLMFinetuneScen"
    """Scenario class for LLM fine-tuning tasks."""

    hypothesis_gen: str = "rdagent.scenarios.finetune.proposal.proposal.LLMFinetuneExpGen"
    """Hypothesis generation class for LLM fine-tuning tasks."""

    hypothesis2experiment: str = "rdagent.scenarios.finetune.proposal.proposal.LLMHypothesis2Experiment"
    """Hypothesis to experiment converter.
    Function: Convert abstract LLM fine-tuning hypotheses into concrete experiment configurations.
    """

    coder: str = "rdagent.components.coder.finetune.LLMFinetuneCoSTEER"
    """Code generator.
    Function: Generate LLM fine-tuning code based on experiment design.
    """

    runner: str = "rdagent.scenarios.finetune.train.runner.LLMFinetuneRunner"
    """Code runner.
    Function: Execute LLM fine-tuning code in a Docker environment.
    """

    summarizer: str = "rdagent.scenarios.finetune.dev.feedback.LLMExperiment2Feedback"
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
    enable_model_dump: bool = True
    app_tpl: str = "scenarios/finetune"

    # Data paths and processing
    local_data_path: str = ""
    file_path: str | None = None  # FT_FILE_PATH/dataset/<dataset>/, FT_FILE_PATH/model/<baseModel>/
    show_nan_columns: bool = False
    sample_data_by_LLM: bool = True

    # LLM-specific fields
    base_model_name: str | None = None
    dataset: str = ""

    @property
    def task(self) -> str:
        """Generate task name using prev_model_dirname function."""
        if self.base_model_name and self.dataset:
            return prev_model_dirname(self.base_model_name, self.dataset)
        return ""


# Global setting instance for LLM finetuning scenario
FT_RD_SETTING = LLMFinetunePropSetting()
