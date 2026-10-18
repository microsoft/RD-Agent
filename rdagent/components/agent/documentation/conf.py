"""Provider selection for pipeline documentation lookup."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    provider: Literal["auto", "parallel", "context7"] = "auto"
    timeout: float = Field(default=120, gt=0)

    model_config = SettingsConfigDict(env_prefix="DOCUMENTATION_SEARCH_")


SETTINGS = Settings()
