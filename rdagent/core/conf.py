from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
)


class ExtendedBaseSettings(BaseSettings):

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # 1) walk from base class
        def base_iter(settings_cls: type[ExtendedBaseSettings]) -> list[type[ExtendedBaseSettings]]:
            bases = []
            for cl in settings_cls.__bases__:
                if issubclass(cl, ExtendedBaseSettings) and cl is not ExtendedBaseSettings:
                    bases.append(cl)
                    bases.extend(base_iter(cl))
            return bases

        # 2) Build EnvSettingsSource from base classes, so we can add parent Env Sources
        parent_env_settings = [
            EnvSettingsSource(
                base_cls,
                case_sensitive=base_cls.model_config.get("case_sensitive"),
                env_prefix=base_cls.model_config.get("env_prefix"),
                env_nested_delimiter=base_cls.model_config.get("env_nested_delimiter"),
            )
            for base_cls in base_iter(cast("type[ExtendedBaseSettings]", settings_cls))
        ]
        return init_settings, env_settings, *parent_env_settings, dotenv_settings, file_secret_settings


class RDAgentSettings(ExtendedBaseSettings):

    # azure document intelligence configs
    azure_document_intelligence_key: str = ""
    azure_document_intelligence_endpoint: str = ""
    # factor extraction conf
    max_input_duplicate_factor_group: int = 300
    max_output_duplicate_factor_group: int = 20
    max_kmeans_group_number: int = 40

    # workspace conf
    workspace_path: Path = Path.cwd() / "git_ignore_folder" / "RD-Agent_workspace"

    # multi processing conf
    multi_proc_n: int = 1

    # pickle cache conf
    cache_with_pickle: bool = True  # whether to use pickle cache
    pickle_cache_folder_path_str: str = str(
        Path.cwd() / "pickle_cache/",
    )  # the path of the folder to store the pickle cache
    use_file_lock: bool = (
        True  # when calling the function with same parameters, whether to use file lock to avoid
        # executing the function multiple times
    )

    # misc
    """The limitation of context stdout"""
    stdout_context_len: int = 400
    stdout_line_len: int = 10000

    enable_mlflow: bool = False

    initial_fator_library_size: int = 20

    # parallel loop
    step_semaphore: int | dict[str, int] = 1
    """the semaphore for each step;  you can specify a overall semaphore
    or a step-wise semaphore like {"coding": 3, "running": 2}"""

    def get_max_parallel(self) -> int:
        """Based on the setting of semaphore, return the maximum number of parallel loops"""
        if isinstance(self.step_semaphore, int):
            return self.step_semaphore
        return max(self.step_semaphore.values())

    # NOTE: for debug
    # the following function only serves as debugging and is necessary in main logic.
    subproc_step: bool = False

    def is_force_subproc(self) -> bool:
        return self.subproc_step or self.get_max_parallel() > 1


RD_AGENT_SETTINGS = RDAgentSettings()
