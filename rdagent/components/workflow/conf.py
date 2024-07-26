from pydantic_settings import BaseSettings


class BasePropSetting(BaseSettings):
    """
    The common part of the config for RD Loop to propose and developement
    You can add following config in the subclass to distinguish the environment variables.

    .. code-block:: python

        class Config:
            env_prefix = "DM_MODEL_"  # Use MODEL_CODER_ as prefix for environment variables
            protected_namespaces = ()  # Add 'model_' to the protected namespaces
    """

    scen: str = ""
    hypothesis_gen: str = ""
    hypothesis2experiment: str = ""
    coder: str = ""
    runner: str = ""
    summarizer: str = ""

    evolving_n: int = 10
