"""
Settings for RAG agent.

TODO: how run the RAG mcp server
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project specific settings."""

    url: str = "http://localhost:8124/mcp"
    timeout: int = 120

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        # extra="allow", # Does it allow extrasettings
    )


SETTINGS = Settings()
