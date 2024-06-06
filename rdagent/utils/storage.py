"""
Concerete storage classes
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fincov2.logging.base import Storage
from fincov2.logging.conf import LogSettings
from utils.mod import import_class

class FileStorage(Storage):
    """
    The info are logginged to the file systems

    TODO: describe the storage format
    """

    FS = None

    def __init__(self, path: str = "./log/") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def log(self, obj: object, name: str = "", obj_type: Literal["json", "text"] = "text") -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        uri_l = name.split(".")
        cur_p = self.path
        for u in uri_l:
            cur_p = cur_p / u
        cur_p.mkdir(parents=True, exist_ok=True)
        path = cur_p / f"{timestamp}.log"
        if obj_type == "json":
            with path.with_suffix(".json").open("w") as f:
                json.dump(json.loads(str(obj)), f)
        else:
            obj = str(obj)
            with path.open("w") as f:
                f.write(obj)

def get_or_create_storage() -> Storage:
    """
    return the storage based on storage
    """
    if FileStorage.FS is None:
        ls = LogSettings()
        fs_cls = import_class(ls.backend_storage)
        FileStorage.FS = fs_cls(**ls.backend_storage_kwargs)
    return FileStorage.FS
