from rdagent.core.conf import ExtendedBaseSettings


class BasePropSetting(ExtendedBaseSettings):
    """
    The common part of the config for RD Loop to propose and development
    You can add following config in the subclass to distinguish the environment variables.
    """

    scen: str | None = None
    knowledge_base: str | None = None
    knowledge_base_path: str | None = None
    hypothesis_gen: str | None = None
    interactor: str | None = None
    hypothesis2experiment: str | None = None
    coder: str | None = None
    runner: str | None = None
    summarizer: str | None = None

    evolving_n: int = 10
