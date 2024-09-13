from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class RunnerSettings(BaseSettings):
    class Config:
        env_prefix = "RUNNER_"  # Use RUNNER_ as prefix for environment variables

    cache_result: bool = True  # whether to cache the result of the docker execution
    cache_path: str = str(Path.cwd() / "runner_cache/")  # the path to store the cache


RUNNER_SETTINGS = RunnerSettings()
