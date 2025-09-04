from pathlib import Path
from typing import Literal

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import (
    CondaConf,
    Env,
    FTDockerEnv,
    LocalEnv,
)


class FTCoderCoSTEERSettings(CoSTEERSettings):
    """LLM Fine-tuning CoSTEER settings"""

    class Config:
        env_prefix = "FT_Coder_CoSTEER_"

    max_seconds_multiplier: int = 8
    """LLM training takes longer, use higher multiplier"""

    env_type: str = "docker"
    """Environment type for LLM fine-tuning (docker/conda)"""

    extra_evaluator: list[str] = ["rdagent.app.finetune.share.eval.PrevModelLoadEvaluator"]
    """LLM-specific evaluators for prev model loading check"""

    extra_eval: list[str] = []
    """Extra evaluators (similar to DS implementation)"""


def _get_standard_ft_volumes() -> dict:
    """Get standard mount volume configuration for LLM finetune environments.

    Creates standard directory mappings:
    - models -> /assets/models (ro)
    - datasets -> /assets/datasets (ro)
    - preprocessed_datasets -> /assets/preprocessed_datasets (ro)

    Returns:
        Dictionary of local_path -> docker_mount_config mappings
    """
    # Import here to avoid circular imports
    from rdagent.app.finetune.llm.conf import FT_RD_SETTING

    base_path = Path(FT_RD_SETTING.file_path)
    volumes = {}

    # Read-only mounts for data and models
    readonly_mounts = [
        ("models", "/assets/models"),
        ("datasets", "/assets/datasets"),
        ("preprocessed_datasets", "/assets/preprocessed_datasets"),
    ]

    for local_dir, docker_path in readonly_mounts:
        local_path = base_path / local_dir
        volumes[str(local_path)] = {"bind": docker_path, "mode": "ro"}

    return volumes


def get_ft_env(
    extra_volumes: dict = {},
    running_timeout_period: int | None = None,
    enable_cache: bool | None = None,
) -> Env:
    """LLM finetune dedicated environment construction function.

    Automatically includes standard finetune volume mounts:
    - models -> /assets/models (ro)
    - datasets -> /assets/datasets (ro)
    - preprocessed_datasets -> /assets/preprocessed_datasets (ro)
    - output -> /workspace/output (rw, auto-created)

    Args:
        extra_volumes: Additional volume mounts beyond standard ones
        running_timeout_period: Timeout period for environment operations
        enable_cache: Whether to enable caching

    Returns:
        Configured environment ready for use
    """
    # Import here to avoid circular imports
    from rdagent.app.finetune.llm.conf import FT_RD_SETTING

    conf = FTCoderCoSTEERSettings()

    # Use default timeout if not provided
    if running_timeout_period is None:
        running_timeout_period = FT_RD_SETTING.debug_timeout

    # Use dedicated LLM docker and conda env
    if conf.env_type == "docker":
        env = FTDockerEnv()
    elif conf.env_type == "conda":
        # Use a dedicated llm conda env name
        # TODO: enable conda environment
        env = LocalEnv(conf=CondaConf(conda_env_name="llm_finetune"))
    else:
        raise ValueError(f"Unknown env type: {conf.env_type}")

    # Combine standard finetune volumes with extra volumes
    standard_volumes = _get_standard_ft_volumes()
    combined_volumes = standard_volumes.copy()
    combined_volumes.update(extra_volumes)

    env.conf.extra_volumes = combined_volumes
    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache
    env.prepare()
    return env


def get_clear_ws_cmd(stage: Literal["before_training", "before_inference"] = "before_training") -> str:
    """
    Clean the files in LLM finetune workspace to a specific stage

    Args:
        stage: Stage to clean to, either "before_training" or "before_inference"

    Returns:
        Command string to clean workspace files
    """
    from rdagent.app.finetune.llm.conf import FT_RD_SETTING

    assert stage in ["before_training", "before_inference"], f"Unknown stage: {stage}"

    if FT_RD_SETTING.enable_model_dump and stage == "before_training":
        # Clean output directory and all training artifacts before training
        cmd = "rm -rf output/ checkpoint-* adapter_* *.safetensors *.bin training_*.json *_metrics.json config.yaml trace.log"
    else:
        # Clean only essential files (keep models for inference)
        cmd = "rm -f training_*.json *_metrics.json config.yaml trace.log"
    return cmd
