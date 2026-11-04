import os
from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import CondaConf, Env, LocalEnv


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


def get_factor_env(
    conf_type: Optional[str] = None,
    extra_volumes: dict = {},
    running_timeout_period: int = 600,
    enable_cache: Optional[bool] = None,
) -> Env:
    conf = FactorCoSTEERSettings()
    if os.environ.get("CONDA_DEFAULT_ENV"):
        env = LocalEnv(conf=(CondaConf(conda_env_name=os.environ.get("CONDA_DEFAULT_ENV"))))
    else:
        # No conda on this host (upstream: hasattr(conf, "python_bin") is always
        # True, so the conda branch crashed with CONDA_DEFAULT_ENV unset).
        # Factor code itself runs via conf.python_bin resolved from the caller's
        # PATH (factor.py), so derive bin_path from that same interpreter to keep
        # runtime_info.py reporting the environment the factor code will see.
        # If python_bin is not on PATH at all (e.g. no bare `python` binary),
        # fall back to the interpreter running rdagent.
        import shutil
        import sys
        from pathlib import Path as _Path

        from rdagent.utils.env import LocalConf

        resolved = shutil.which(conf.python_bin)
        bin_dir = _Path(resolved).parent if resolved else _Path(sys.executable).parent
        env = LocalEnv(conf=LocalConf(bin_path=str(bin_dir), default_entry="python main.py"))
    env.conf.extra_volumes = extra_volumes.copy()
    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache
    env.prepare()
    return env


FACTOR_COSTEER_SETTINGS = FactorCoSTEERSettings()
