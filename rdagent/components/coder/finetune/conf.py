from pathlib import Path
from typing import Literal

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import (
    CondaConf,
    Env,
    FTDockerEnv,
    LocalEnv,
)

FT_YAML_FILE_NAME = "train.yaml"
FT_DEBUG_YAML_FILE_NAME = "debug_train.yaml"


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
    """Extra evaluators"""


def _get_standard_ft_volumes() -> dict:
    """Get standard mount volume configuration for LLM finetune environments.

    Creates standard directory mappings:
    - models -> /assets/models (ro)
    - datasets -> /assets/datasets (ro)

    Returns:
        Dictionary of local_path -> docker_mount_config mappings
    """
    base_path = Path(FT_RD_SETTING.file_path)
    volumes = {}

    # Read-only mounts for data and models
    readonly_mounts = [
        ("models", "/assets/models"),
        ("datasets", "/assets/datasets"),
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
    - output -> /workspace/output (rw, auto-created)

    Note: .llama_factory_info is no longer automatically mounted.
    Pass llama_factory_info volume via extra_volumes when needed.

    Args:
        extra_volumes: Additional volume mounts beyond standard ones
        running_timeout_period: Timeout period for environment operations
        enable_cache: Whether to enable caching (None means use config value)

    Returns:
        Configured environment ready for use
    """

    conf = FTCoderCoSTEERSettings()

    # Use default timeout if not provided
    if running_timeout_period is None:
        running_timeout_period = FT_RD_SETTING.debug_timeout

    # Use config value if enable_cache is not explicitly provided
    if enable_cache is None:
        enable_cache = FT_RD_SETTING.docker_enable_cache

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
    assert stage in ["before_training", "before_inference"], f"Unknown stage: {stage}"

    if stage == "before_training":
        # Clean all training outputs before new training
        cmd = f"rm -rf output/ checkpoint-* adapter_* *.safetensors *.bin training_*.json *_metrics.json {FT_YAML_FILE_NAME} trace.log"
    else:
        # Clean only logs before inference (keep model outputs)
        cmd = f"rm -f training_*.json *_metrics.json {FT_YAML_FILE_NAME} trace.log"
    return cmd
