"""
The context7 is based on a modified version of the context7.

You can follow the instructions to install it

    mkdir -p ~/tmp/
    cd ~/tmp/ && git clone https://github.com/Hoder-zyf/context7.git
    cd ~/tmp/context7
    npm install -g bun
    bun i && bun run build
    bun run dist/index.js --transport http --port 8123 # > bun.out 2>&1 &
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project specific settings."""

    url: str = "http://localhost:8123/mcp"
    timeout: int = 120

    model_config = SettingsConfigDict(
        env_prefix="CONTEXT7_",
        # extra="allow", # Does it allow extrasettings
    )


SETTINGS = Settings()
