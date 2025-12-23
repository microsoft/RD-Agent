import os
from pathlib import Path
from typing import Literal

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import (
    BenchmarkCondaConf,
    BenchmarkCondaEnv,
    BenchmarkDockerConf,
    BenchmarkDockerEnv,
    DockerEnv,
    Env,
    FTCondaConf,
    FTCondaEnv,
    FTDockerEnv,
)


def is_docker_env(env: Env) -> bool:
    """Check if the environment is Docker-based."""
    return isinstance(env, DockerEnv)


def get_workspace_prefix(env: Env) -> str:
    """Return workspace path prefix based on env type.

    Docker uses /workspace as mount point, conda uses current directory.
    """
    return "/workspace" if is_docker_env(env) else "."


FT_YAML_FILE_NAME = "train.yaml"
FT_DATA_PROC_FILE_NAME = "data_process.py"
FT_DEBUG_YAML_FILE_NAME = "debug_train.yaml"
FT_DATA_FILE_NAME = "data.json"
FT_DATA_SCRIPT_NAME = "process_data.py"

# ENV Info:  the path of the model and dataset in the container/environment
FT_MODEL_PATH = "/assets/models"
FT_DATASET_PATH = "/assets/datasets"


class FTPathConfig:
    """Centralized path configuration for FT scenario.

    Provides environment-aware paths for Docker vs Conda modes.
    Uses lazy evaluation (properties) to avoid import-time errors.

    Usage:
        from rdagent.components.coder.finetune.conf import FT_PATHS

        models_path = FT_PATHS.models      # e.g., "/assets/models/" or "/path/to/finetune/models/"
        datasets_path = FT_PATHS.datasets  # e.g., "/assets/datasets/" or "/path/to/finetune/datasets/"
        workspace_path = FT_PATHS.workspace  # e.g., "/workspace/" or "./"
    """

    @property
    def is_docker(self) -> bool:
        """Check if current environment is Docker-based."""
        return FTCoderCoSTEERSettings().env_type == "docker"

    @property
    def models(self) -> str:
        """Model directory path (with trailing slash)."""
        if self.is_docker:
            return FT_MODEL_PATH + "/"
        return str(FT_RD_SETTING.file_path / "models") + "/"

    @property
    def datasets(self) -> str:
        """Dataset directory path for raw datasets (with trailing slash)."""
        if self.is_docker:
            return FT_DATASET_PATH + "/"
        return str(FT_RD_SETTING.file_path / "datasets") + "/"

    @property
    def workspace(self) -> str:
        """Workspace path prefix for prompts (with trailing slash)."""
        return "/workspace/" if self.is_docker else "./"

    @property
    def deepspeed(self) -> str:
        """DeepSpeed config directory."""
        if self.is_docker:
            return "/app/examples/deepspeed/"
        # Conda mode: use bundled deepspeed configs in project
        # Path: conf.py -> finetune -> coder -> components -> rdagent -> scenarios/finetune/deepspeed
        rdagent_root = Path(__file__).parent.parent.parent.parent
        deepspeed_path = rdagent_root / "scenarios" / "finetune" / "deepspeed"
        return str(deepspeed_path) + "/" if deepspeed_path.exists() else ""


# Singleton instance for path configuration
FT_PATHS = FTPathConfig()


class FTCoderCoSTEERSettings(CoSTEERSettings):
    """LLM Fine-tuning CoSTEER settings"""

    class Config:
        env_prefix = "FT_Coder_CoSTEER_"

    max_seconds_multiplier: int = 8
    """LLM training takes longer, use higher multiplier"""

    env_type: str = "docker"
    """Environment type for LLM fine-tuning (docker/conda)"""

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
        ("models", FT_MODEL_PATH),
        ("datasets", FT_DATASET_PATH),
    ]

    for local_dir, docker_path in readonly_mounts:
        local_path = base_path / local_dir
        volumes[str(local_path)] = {"bind": docker_path, "mode": "ro"}

    return volumes


def get_ft_env(
    extra_volumes: dict = {},
    operation: str = "full_training",
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
        operation: Operation type for timeout selection.
            - "data_processing": Data processing (data_processing_timeout)
            - "micro_batch": Micro-batch test (micro_batch_timeout)
            - "full_training": Full training (full_timeout)
        enable_cache: Whether to enable caching (None means use config value)

    Returns:
        Configured environment ready for use
    """

    conf = FTCoderCoSTEERSettings()

    # Select timeout based on operation type
    timeout_map = {
        "data_processing": FT_RD_SETTING.data_processing_timeout,
        "micro_batch": FT_RD_SETTING.micro_batch_timeout,
        "full_training": FT_RD_SETTING.full_timeout,
    }
    running_timeout_period = timeout_map.get(operation, FT_RD_SETTING.full_timeout)

    # Use config value if enable_cache is not explicitly provided
    if enable_cache is None:
        enable_cache = FT_RD_SETTING.docker_enable_cache

    # Use dedicated LLM docker or conda env based on config
    if conf.env_type == "docker":
        env = FTDockerEnv()
        # Docker mode: setup volume mounts for models/datasets
        standard_volumes = _get_standard_ft_volumes()
        combined_volumes = standard_volumes.copy()
        combined_volumes.update(extra_volumes)
        env.conf.extra_volumes = combined_volumes
    elif conf.env_type == "conda":
        env = FTCondaEnv(conf=FTCondaConf())  # Auto-installs dependencies if env doesn't exist
        # Conda mode: no volume mounts needed, use local paths directly
        # extra_volumes are ignored in conda mode
    else:
        raise ValueError(f"Unknown env type: {conf.env_type}")

    env.conf.running_timeout_period = running_timeout_period
    env.conf.enable_cache = enable_cache
    env.prepare()
    return env


def get_data_processing_env(
    enable_cache: bool | None = None,
) -> tuple[Env, dict]:
    """Get environment for data processing scripts with LLM API access.

    This environment is configured for running data processing scripts that may
    need to call LLM APIs. It includes:
    - Standard finetune volume mounts (datasets, models)
    - LLM API environment variables (OPENAI_API_KEY, OPENAI_BASE_URL, etc.)

    Args:
        enable_cache: Whether to enable Docker caching

    Returns:
        Tuple of (env, env_vars) where env_vars contains LLM API keys
        to be passed to env.run() as the env parameter
    """
    env = get_ft_env(
        operation="data_processing",
        enable_cache=enable_cache,
    )

    # Collect LLM API environment variables to pass to env.run()
    llm_env_vars = {"PYTHONPATH": "./"}  # Base env var

    # Pass OPENAI_API_KEY directly
    if api_key := os.getenv("OPENAI_API_KEY"):
        llm_env_vars["OPENAI_API_KEY"] = api_key

    # Read OPENAI_API_BASE from env, but pass as OPENAI_BASE_URL (OpenAI SDK expects this name)
    if api_base := os.getenv("OPENAI_API_BASE"):
        llm_env_vars["OPENAI_BASE_URL"] = api_base

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


def get_benchmark_env(
    extra_volumes: dict = {},
    timeout: int | None = None,
) -> Env:
    """OpenCompass benchmark environment construction function.

    Supports both Docker and conda environments based on FT_Coder_CoSTEER_env_type.

    Args:
        extra_volumes: Additional volume mounts (only used in Docker mode)
        timeout: Running timeout in seconds (None uses config default)

    Returns:
        Configured environment ready for benchmark evaluation
    """
    conf = FTCoderCoSTEERSettings()

    # Use benchmark-specific timeout or config default
    if timeout is None:
        # 0 means no timeout, use 7 days as practical "infinite"
        timeout = FT_RD_SETTING.benchmark_timeout if FT_RD_SETTING.benchmark_timeout > 0 else 86400 * 7

    if conf.env_type == "docker":
        docker_conf = BenchmarkDockerConf()
        docker_conf.running_timeout_period = timeout

        # Setup finetune share folder mount for models
        benchmark_volumes = {}
        try:
            (FT_RD_SETTING.file_path / "benchmarks").mkdir(parents=True, exist_ok=True)
            benchmark_volumes[str(FT_RD_SETTING.file_path.resolve())] = {"bind": "/finetune", "mode": "rw"}
            benchmark_volumes[str((FT_RD_SETTING.file_path / "benchmarks").resolve())] = {
                "bind": "/benchmarks",
                "mode": "rw",
            }
        except (PermissionError, OSError):
            pass

        benchmark_volumes.update(extra_volumes)
        docker_conf.extra_volumes = benchmark_volumes

        env = BenchmarkDockerEnv(conf=docker_conf)
    elif conf.env_type == "conda":
        conda_conf = BenchmarkCondaConf()
        conda_conf.running_timeout_period = timeout
        env = BenchmarkCondaEnv(conf=conda_conf)  # Auto-installs dependencies if env doesn't exist
    else:
        raise ValueError(f"Unknown env type: {conf.env_type}")

    env.prepare()
    return env
