"""
AutoRL-Bench 配置

独立配置，不依赖 RL_RD_SETTING，只复用 rdagent 基类。
"""

from pathlib import Path

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class AutoRLBenchSettings(ExtendedBaseSettings):
    """AutoRL-Bench 配置

    环境变量前缀: AUTORL_
    例如: AUTORL_FILE_PATH=/data/autorl_bench
    """

    model_config = SettingsConfigDict(env_prefix="AUTORL_", protected_namespaces=())

    file_path: Path = Path.cwd() / "git_ignore_folder" / "rl_files"
    rdagent_root: Path = Path.cwd()  # Docker 挂载用，可通过 AUTORL_RDAGENT_ROOT 覆盖


AUTORL_BENCH_SETTING = AutoRLBenchSettings()


def get_autorl_bench_dir() -> Path:
    return Path(__file__).parent


def get_workspace_dir() -> Path:
    return get_autorl_bench_dir() / "workspace"


def get_instructions_file() -> Path:
    return get_autorl_bench_dir() / "core" / "instructions.md"


def get_models_dir() -> Path:
    return AUTORL_BENCH_SETTING.file_path / "models"


def get_data_dir() -> Path:
    return AUTORL_BENCH_SETTING.file_path / "datasets"


def get_baseline_cache_dir() -> Path:
    return AUTORL_BENCH_SETTING.file_path / "baseline_workspace"
