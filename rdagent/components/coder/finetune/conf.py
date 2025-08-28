from typing import Literal

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import (
    CondaConf,
    DockerEnv,
    Env,
    LLMDockerConf,
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


def get_ft_env(
    extra_volumes: dict = {},
    running_timeout_period: int | None = None,
    enable_cache: bool | None = None,
) -> Env:
    """LLM finetune dedicated environment construction function.

    - Create Docker or Conda environment according to configuration
    - Configure mount volumes (extra_volumes)
    - Set running timeout/cache
    - Call env.prepare() to make the environment ready
    """
    # Import here to avoid circular imports
    from rdagent.app.finetune.llm.conf import FT_RD_SETTING

    conf = FTCoderCoSTEERSettings()

    # Use default timeout if not provided
    if running_timeout_period is None:
        running_timeout_period = FT_RD_SETTING.debug_timeout

    # Use dedicated LLM docker and conda env
    if conf.env_type == "docker":
        env = DockerEnv(conf=LLMDockerConf())
    elif conf.env_type == "conda":
        # Use a dedicated llm conda env name
        env = LocalEnv(conf=CondaConf(conda_env_name="llm_finetune"))
    else:
        raise ValueError(f"Unknown env type: {conf.env_type}")

    env.conf.extra_volumes = extra_volumes.copy()
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
