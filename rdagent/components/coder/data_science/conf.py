from typing import Literal

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import (
    CondaConf,
    DockerEnv,
    DSDockerConf,
    Env,
    LocalEnv,
    MLEBDockerConf,
    MLECondaConf,
)


class DSCoderCoSTEERSettings(CoSTEERSettings):
    """Data Science CoSTEER settings"""

    class Config:
        env_prefix = "DS_Coder_CoSTEER_"

    max_seconds: int = 2400
    env_type: str = "docker"
    # TODO: extract a function for env and conf.


def get_ds_env(
    conf_type: Literal["kaggle", "mlebench"] = "kaggle",
    extra_volumes: dict = {},
    running_timeout_period: int = (
        DS_RD_SETTING.debug_timeout if not DS_RD_SETTING.sample_data_by_LLM else DS_RD_SETTING.full_timeout
    ),
) -> Env:
    """
    Retrieve the appropriate environment configuration based on the env_type setting.

    Returns:
        Env: An instance of the environment configured either as DockerEnv or LocalEnv.

    Raises:
        ValueError: If the env_type is not recognized.
    """
    conf = DSCoderCoSTEERSettings()
    assert conf_type in ["kaggle", "mlebench"], f"Unknown conf_type: {conf_type}"

    if conf.env_type == "docker":
        env_conf = DSDockerConf() if conf_type == "kaggle" else MLEBDockerConf()
        env = DockerEnv(conf=env_conf)
    elif conf.env_type == "conda":
        env = LocalEnv(
            conf=(
                CondaConf(conda_env_name=conf_type) if conf_type == "kaggle" else MLECondaConf(conda_env_name=conf_type)
            )
        )
    else:
        raise ValueError(f"Unknown env type: {conf.env_type}")
    env.conf.extra_volumes = extra_volumes
    env.conf.running_timeout_period = running_timeout_period
    return env


def get_clear_ws_cmd(stage: Literal["before_training", "before_inference"] = "before_training") -> str:
    """
    Clean the files in workspace to a specific stage
    """
    assert stage in ["before_training", "before_inference"], f"Unknown stage: {stage}"
    if DS_RD_SETTING.enable_model_dump and stage == "before_training":
        cmd = "rm -r submission.csv scores.csv models"
    else:
        cmd = "rm submission.csv scores.csv"
    return cmd
