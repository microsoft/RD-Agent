from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class LogSettings(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOG_", protected_namespaces=())

    trace_path: str = str(Path.cwd() / "log" / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f"))

    format_console: str | None = None
    """"If it is None, leave it as the default"""

    ui_server_port: int | None = None

    storages: dict[str, list[int | str]] = {}

    def model_post_init(self, _context: Any, /) -> None:
        if self.ui_server_port is not None:
            self.storages["rdagent.log.ui.storage.WebStorage"] = [self.ui_server_port, self.trace_path]


LOG_SETTINGS = LogSettings()
