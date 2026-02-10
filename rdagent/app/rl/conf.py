from pathlib import Path

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class RLPostTrainingPropSetting(ExtendedBaseSettings):
    """RL Post-training dedicated property settings.

    Use RL_ env prefix for overrides.
    """

    model_config = SettingsConfigDict(env_prefix="RL_", protected_namespaces=())

    # Main Components
    scen: str = "rdagent.scenarios.rl.scen.scenario.RLPostTrainingScen"
    hypothesis_gen: str = "rdagent.scenarios.rl.proposal.proposal.RLPostTrainingExpGen"
    coder: str = "rdagent.components.coder.rl.RLCoSTEER"
    runner: str = "rdagent.scenarios.rl.train.runner.RLPostTrainingRunner"
    summarizer: str = "rdagent.scenarios.rl.dev.feedback.RLExperiment2Feedback"

    # Resource paths (unified directory management, similar to SFT)
    file_path: Path = Path.cwd() / "git_ignore_folder" / "rl_files"
    """RL resource root directory. Contains datasets/ and models/ subdirectories.
    Can be overridden via RL_FILE_PATH environment variable."""

    # Core config
    base_model: str | None = None
    """Model name (e.g., 'Qwen2.5-Coder-0.5B-Instruct'). Docker path: /models/{base_model}"""

    benchmark: str | None = None
    """Benchmark/dataset name (e.g., 'gsm8k'). Docker path: /data/{benchmark}"""

    # Benchmark evaluation
    benchmark_timeout: int = 0
    """Benchmark evaluation timeout in seconds. 0 means no timeout."""


# Global setting instance
RL_RD_SETTING = RLPostTrainingPropSetting()
