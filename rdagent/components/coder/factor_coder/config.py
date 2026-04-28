import os
from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import CondaConf, Env, LocalEnv, QTDockerEnv


class FactorCoSTEERSettings(CoSTEERSettings):
    model_config = SettingsConfigDict(env_prefix="FACTOR_CoSTEER_")

    data_folder: str = "git_ignore_folder/factor_implementation_source_data"
    """Path to the folder containing financial data (default is fundamental data in Qlib)"""

    data_folder_debug: str = "git_ignore_folder/factor_implementation_source_data_debug"
    """Path to the folder containing partial financial data (for debugging)"""

    simple_background: bool = False
    """Whether to use simple background information for code feedback"""

    file_based_execution_timeout: int = 3600
    """Timeout in seconds for each factor implementation execution"""

    select_method: str = "random"
    """Method for the selection of factors implementation"""

    python_bin: str = "python"
    """Path to the Python binary"""

    env_type: str = "conda"
    """Environment type: 'conda' for LocalEnv with Conda, 'docker' for QTDockerEnv"""


def get_factor_env(
    conf_type: Optional[str] = None,
    extra_volumes: dict = {},
    running_timeout_period: int = 600,
    enable_cache: Optional[bool] = None,
) -> Env:
    conf = FactorCoSTEERSettings()

    env_type = os.environ.get("MODEL_COSTEER_ENV_TYPE", conf.env_type)

    if env_type == "docker":
        env = QTDockerEnv()
        if extra_volumes:
            env.conf.extra_volumes.update(extra_volumes)
    else:
        conda_env_name = os.environ.get("CONDA_DEFAULT_ENV", "rdagent")
        env = LocalEnv(conf=(CondaConf(conda_env_name=conda_env_name)))
        env.conf.extra_volumes = extra_volumes.copy()

    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache
    env.prepare()
    return env


FACTOR_COSTEER_SETTINGS = FactorCoSTEERSettings()
