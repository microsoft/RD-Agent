import os

from pydantic_settings import SettingsConfigDict

from rdagent.components.coder.data_science.conf import DSCoderCoSTEERSettings
from rdagent.core.conf import RD_AGENT_SETTINGS, ExtendedBaseSettings
from rdagent.scenarios.finetune.utils import prev_model_dirname
from rdagent.utils.env import (
    CondaConf,
    DockerEnv,
    Env,
    LLMDockerConf,
    LocalEnv,
)


class LLMFinetunePropSetting(ExtendedBaseSettings):
    """LLM Fine-tune dedicated property settings.

    - Adjust timeouts and template
    - Use FT_ env prefix for overrides
    """

    model_config = SettingsConfigDict(env_prefix="FT_", protected_namespaces=())

    # Core Components
    scen: str = "rdagent.scenarios.finetune.scen.LLMFinetuneScen"
    hypothesis_gen: str = "rdagent.app.finetune.llm.proposal.FinetuneExpGen"

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


def get_ft_env(
    extra_volumes: dict = {},
    running_timeout_period: int | None = FT_RD_SETTING.debug_timeout,
    enable_cache: bool | None = None,
) -> Env:
    """LLM finetune dedicated environment construction function, equivalent to the responsibility of get_ds_env.

    - Create Docker or Conda environment according to configuration
    - Configure mount volumes (extra_volumes)
    - Set running timeout/cache
    - Call env.prepare() to make the environment ready
    """
    conf = DSCoderCoSTEERSettings()

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


def update_settings(
    dataset: str,
    model: str,
):
    """
    Update the RD_AGENT_SETTINGS with the values from LLM_FINETUNE_SETTINGS.
    """
    RD_AGENT_SETTINGS.app_tpl = FT_RD_SETTING.app_tpl
    os.environ["DS_CODER_COSTEER_EXTRA_EVALUATOR"] = '["rdagent.app.finetune.share.eval.PrevModelLoadEvaluator"]'
    # Update FT setting instance (use dataset instead of competition in LLM loop)
    FT_RD_SETTING.dataset = dataset
    FT_RD_SETTING.base_model_name = model
    # If FT_FILE_PATH is configured, directly mount from its dataset directory
    if FT_RD_SETTING.file_path:
        FT_RD_SETTING.local_data_path = os.path.join(FT_RD_SETTING.file_path, "dataset")
