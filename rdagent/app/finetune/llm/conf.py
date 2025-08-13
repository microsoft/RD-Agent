import os
import shutil

from pydantic_settings import SettingsConfigDict

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.conf import DSCoderCoSTEERSettings
from rdagent.core.conf import RD_AGENT_SETTINGS, ExtendedBaseSettings
from rdagent.utils.env import (
    CondaConf,
    DockerEnv,
    DSDockerConf,
    Env,
    LocalEnv,
)


class LLMFinetuneScen(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="FT_", protected_namespaces=())
    scen: str = "rdagent.app.finetune.llm.scen.LLMFinetuneScen"
    """
    Scenario class for data science tasks.
    - For Kaggle competitions, use: "rdagent.scenarios.data_science.scen.KaggleScen"
    - For custom data science scenarios, use: "rdagent.scenarios.data_science.scen.DataScienceScen"
    - For LLM finetune scenarios, use: "rdagent.app.finetune.llm.scen.LLMFinetuneScen"
    - For Data science finetune scenarios, use: "rdagent.app.finetune.data_science.scen.DSFinetuneScen"
    """

    hypothesis_gen: str = "rdagent.app.finetune.llm.proposal.FinetuneExpGen"
    """Hypothesis generation class"""

    debug_timeout: int = 36000
    """The timeout limit for running on debugging data"""
    full_timeout: int = 360000
    """The timeout limit for running on full data"""

    coder_on_whole_pipeline: bool = True
    enable_model_dump: bool = True
    app_tpl: str = "app/finetune/llm/tpl"

    # Base directory for finetune workspace declared in `.env` as FT_FILE_PATH
    # Expected structure under this directory:
    #   dataset/<name of dataset>/
    #   model/<name of baseModel>/
    #   prev_model/<baseModel_dataset>/
    file_path: str | None = None


def get_ft_env(
    extra_volumes: dict = {},
    running_timeout_period: int | None = DS_RD_SETTING.debug_timeout,
    enable_cache: bool | None = None,
) -> Env:
    """LLM finetune dedicated environment construction function, equivalent to the responsibility of get_ds_env.

    - Create Docker or Conda environment according to configuration
    - Configure mount volumes (extra_volumes)
    - Set running timeout/cache
    - Call env.prepare() to make the environment ready
    """
    conf = DSCoderCoSTEERSettings()

    # TODO: add a dedicated llm docker and conda env
    if conf.env_type == "docker":
        env = DockerEnv(conf=DSDockerConf())
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
):  # TODO: add model, change competition to model_dataset(qizhengli)
    """
    Update the RD_AGENT_SETTINGS with the values from LLM_FINETUNE_SETTINGS.
    """
    LLM_FINETUNE_SETTINGS = LLMFinetuneScen()
    RD_AGENT_SETTINGS.app_tpl = LLM_FINETUNE_SETTINGS.app_tpl
    os.environ["DS_CODER_COSTEER_EXTRA_EVALUATOR"] = '["rdagent.app.finetune.share.eval.PrevModelLoadEvaluator"]'
    for field_name, new_value in LLM_FINETUNE_SETTINGS.model_dump().items():
        if hasattr(DS_RD_SETTING, field_name):
            setattr(DS_RD_SETTING, field_name, new_value)
    # If FT_FILE_PATH is configured, directly mount from its dataset directory
    if LLM_FINETUNE_SETTINGS.file_path:
        DS_RD_SETTING.local_data_path = os.path.join(LLM_FINETUNE_SETTINGS.file_path, "dataset")
    # Downstream still uses the competition field; dataset is in the form "<baseModel>_<dataset>"
    DS_RD_SETTING.competition = dataset
