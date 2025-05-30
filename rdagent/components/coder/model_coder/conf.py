from pydantic_settings import SettingsConfigDict

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings


class ModelCoSTEERSettings(CoSTEERSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_CoSTEER_")

    env_type: str = "conda"  # or "docker"
    """Environment to run model code in coder and runner: 'conda' for local conda env, 'docker' for Docker container"""


MODEL_COSTEER_SETTINGS = ModelCoSTEERSettings()
