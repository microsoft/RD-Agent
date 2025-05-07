from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class UIBasePropSetting(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="UI_", protected_namespaces=())

    default_log_folders: list[str] = ["./log"]

    baseline_result_path: str = "./baseline.csv"

    aide_path: str = "./aide"


UI_SETTING = UIBasePropSetting()
