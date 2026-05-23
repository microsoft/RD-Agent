from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gateway_host: str = "0.0.0.0"
    gateway_port: int = 6900
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    app_version: str = "0.2.0"

    bybit_testnet: bool = True
    bybit_api_key: str = ""
    bybit_api_secret: str = ""

    trace_folder: Path = _repo_root() / "git_ignore_folder" / "traces"
    workspace_path: Path = _repo_root() / "git_ignore_folder" / "RD-Agent_workspace"
    repo_root: Path = _repo_root()

    @property
    def ui_server_port(self) -> int:
        return self.gateway_port


settings = Settings()
