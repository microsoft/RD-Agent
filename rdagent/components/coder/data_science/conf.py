from typing import Literal

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


def get_ds_env(conf_type: Literal["kaggle", "mlebench"] = "kaggle") -> Env:
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
    return env
