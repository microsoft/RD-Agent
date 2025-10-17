from pydantic_settings import BaseSettings, SettingsConfigDict


class DistSettings(BaseSettings):
    """Distributional Agents Settings."""

    host: str = "localhost"
    port: int = 5321

    model_config = SettingsConfigDict(
        env_prefix="DIST_",
        # extra="allow", # Does it allow extrasettings
    )

DIST_SETTING = DistSettings()
