from __future__ import annotations

# TODO: use pydantic for other modules in Qlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class ExtendedEnvSettingsSource(EnvSettingsSource):
    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # Dynamically gather prefixes from the current and parent classes
        prefixes = [self.config.get("env_prefix", "")]
        if hasattr(self.settings_cls, "__bases__"):
            for base in self.settings_cls.__bases__:
                if hasattr(base, "model_config"):
                    parent_prefix = base.model_config.get("env_prefix")
                    if parent_prefix and parent_prefix not in prefixes:
                        prefixes.append(parent_prefix)
        for prefix in prefixes:
            self.env_prefix = prefix
            env_val, field_key, value_is_complex = super().get_field_value(field, field_name)
            if env_val is not None:
                return env_val, field_key, value_is_complex

        return super().get_field_value(field, field_name)


class ExtendedSettingsConfigDict(SettingsConfigDict, total=False): ...


class ExtendedBaseSettings(BaseSettings):

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,  # noqa
        env_settings: PydanticBaseSettingsSource,  # noqa
        dotenv_settings: PydanticBaseSettingsSource,  # noqa
        file_secret_settings: PydanticBaseSettingsSource,  # noqa
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (ExtendedEnvSettingsSource(settings_cls),)


class RDAgentSettings(ExtendedBaseSettings):
    # TODO: (xiao) I think LLMSetting may be a better name.
    # TODO: (xiao) I think most of the config should be in oai.config
    # Log configs
    # TODO: (xiao) think it can be a separate config.
    log_trace_path: str | None = None

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


RD_AGENT_SETTINGS = RDAgentSettings()
