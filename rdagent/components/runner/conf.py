from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# make sure that env variable is loaded while calling Config()
load_dotenv(verbose=True, override=True)

from pydantic_settings import BaseSettings


class RunnerSettings(BaseSettings):
    runner_cache_result: bool = True  # whether to cache the result of the docker execution
    runner_cache_path: str = str(Path.cwd() / "runner_cache/")  # the path to store the cache


RUNNER_SETTINGS = RunnerSettings()
