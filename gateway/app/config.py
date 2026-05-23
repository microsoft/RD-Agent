from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gateway_host: str = "0.0.0.0"
    gateway_port: int = 6900
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    app_version: str = "0.1.0"

    bybit_testnet: bool = True
    bybit_api_key: str = ""
    bybit_api_secret: str = ""


settings = Settings()
