from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class UIBasePropSetting(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="UI_", protected_namespaces=())

    default_log_folders: list[str] = ["./log"]

    baseline_result_path: str = "./baseline.csv"

    aide_path: str = "./aide"

    amlt_path: str = "/data/share_folder_local/amlt"

    static_path: str = "./git_ignore_folder/static"

    trace_folder: str = "./traces"


UI_SETTING = UIBasePropSetting()
