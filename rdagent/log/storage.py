import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .base import Storage


class FileStorage(Storage):
    """
    The info are logginged to the file systems

    TODO: describe the storage format
    """

    def __init__(self, path: str = "./log/") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        obj: object,
        name: str = "",
        save_type: Literal["json", "text", "pkl"] = "text",
        timestamp: datetime | None = None,
    ) -> Path:
        # TODO: We can remove the timestamp after we implement PipeLog
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)

        cur_p = self.path / name.replace(".", "/")
        cur_p.mkdir(parents=True, exist_ok=True)

        path = cur_p / f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S-%f')}.log"

        if save_type == "json":
            path = path.with_suffix(".json")
            with path.open("w") as f:
                try:
                    json.dump(obj, f)
                except TypeError:
                    json.dump(json.loads(str(obj)), f)
            return path
        elif save_type == "pkl":
            path = path.with_suffix(".pkl")
            with path.open("wb") as f:
                pickle.dump(obj, f)
            return path
        elif save_type == "text":
            obj = str(obj)
            with path.open("w") as f:
                f.write(obj)
            return path
