from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import Env, QlibCondaConf, QlibCondaEnv, QTDockerEnv


class ModelCoSTEERSettings(CoSTEERSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_CoSTEER_")

    env_type: str = "conda"  # or "docker"
    """Environment to run model code in coder and runner: 'conda' for local conda env, 'docker' for Docker container"""


def get_model_env(
    conf_type: Optional[str] = None,
    extra_volumes: dict = {},
    running_timeout_period: int = 600,
    enable_cache: Optional[bool] = None,
) -> Env:
    conf = ModelCoSTEERSettings()
    if conf.env_type == "docker":
        env = QTDockerEnv()
    elif conf.env_type == "conda":
        env = QlibCondaEnv(conf=QlibCondaConf())
    else:
        raise ValueError(f"Unknown env type: {conf.env_type}")

    env.conf.extra_volumes = extra_volumes.copy()
    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache
    env.prepare()
    return env


MODEL_COSTEER_SETTINGS = ModelCoSTEERSettings()
