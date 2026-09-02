from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class UIBasePropSetting(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="UI_", protected_namespaces=())

    default_log_folders: list[str] = ["./log"]

    baseline_result_path: str = "./baseline.csv"

    aide_path: str = "./aide"

    amlt_path: str = "/data/share_folder_local/amlt"

    static_path: str = "./git_ignore_folder/static"

    trace_folder: str = "./git_ignore_folder/traces"

    upload_folder: str = "./git_ignore_folder/uploads"

    server_host: str = "127.0.0.1"

    server_auth_token: str = ""

    cors_allowed_origins: list[str] = []

    max_upload_mb: int = 20

    load_legacy_pickle_traces: bool = False

    enable_cache: bool = True


UI_SETTING = UIBasePropSetting()
