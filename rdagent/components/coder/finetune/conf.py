import os
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
FT_DATA_PROC_FILE_NAME = "data_process.py"
FT_DEBUG_YAML_FILE_NAME = "debug_train.yaml"
FT_DATA_FILE_NAME = "data.json"
FT_DATA_SCRIPT_NAME = "process_data.py"

# ENV Info:  the path of the model and dataset in the container/environment
FT_MODEL_PATH = "/assets/models"
FT_DATASET_PATH = "/assets/datasets"


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

    # data related.
    api_base: str = "http://ep14.213428.xyz:38833"
    api_key: str = "sk-1234"
    available_api_models: str = "gpt-4o"


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
        ("models", FT_MODEL_PATH),
        ("datasets", FT_DATASET_PATH),
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

    assert conf.env_type == "docker", f"LLM finetune only supports docker env, got: {conf.env_type}"
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


def get_data_processing_env(
    running_timeout_period: int = 3600,
    enable_cache: bool | None = None,
) -> tuple[Env, dict]:
    """Get environment for data processing scripts with LLM API access.

    This environment is configured for running data processing scripts that may
    need to call LLM APIs. It includes:
    - Standard finetune volume mounts (datasets, models)
    - LLM API environment variables (OPENAI_API_KEY, OPENAI_API_BASE, etc.)

    Args:
        running_timeout_period: Timeout for script execution (default 1 hour)
        enable_cache: Whether to enable Docker caching

    Returns:
        Tuple of (env, env_vars) where env_vars contains LLM API keys
        to be passed to env.run() as the env parameter
    """
    env = get_ft_env(
        running_timeout_period=running_timeout_period,
        enable_cache=enable_cache,
    )

    # Collect LLM API environment variables to pass to env.run()
    llm_env_vars = {"PYTHONPATH": "./"}  # Base env var
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_API_BASE"
    ]:
        value = os.getenv(key)
        if value:
            llm_env_vars[key] = value

    return env, llm_env_vars


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
