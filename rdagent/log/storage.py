import dataclasses
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Literal

from .base import Message, Storage
from .utils import gen_datetime

LOG_LEVEL = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

try:
    import numpy as np
except Exception:  # pragma: no cover - optional
    np = None


def _to_jsonable(obj: object, seen: set[int] | None = None) -> object:
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return "<recursion>"
    seen.add(obj_id)

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.decode("utf-8", errors="replace")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name), seen) for f in dataclasses.fields(obj)}
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump(mode="json"), seen)
        except Exception:
            try:
                return _to_jsonable(obj.model_dump(), seen)
            except Exception:
                pass
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v, seen) for v in obj]
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            try:
                return obj.item()
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        return {"__type__": type(obj).__name__, "__dict__": _to_jsonable(obj.__dict__, seen)}
    return {"__type__": type(obj).__name__, "__repr__": repr(obj)}


def _remove_empty_dir(path: Path) -> None:
    """
    Recursively remove empty directories.
    This function will remove the directory if it is empty after removing its subdirectories.
    """
    if path.is_dir():
        sub_dirs = [sub for sub in path.iterdir() if sub.is_dir()]
        for sub in sub_dirs:
            _remove_empty_dir(sub)

        if not any(path.iterdir()):
            path.rmdir()


class FileStorage(Storage):
    """
    The info are logginged to the file systems

    TODO: describe the storage format
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(
        self,
        obj: object,
        tag: str = "",
        timestamp: datetime | None = None,
        save_type: Literal["json", "text", "pkl"] = "pkl",
        **kwargs: Any,
    ) -> str | Path:
        # TODO: We can remove the timestamp after we implement PipeLog
        timestamp = gen_datetime(timestamp)

        cur_p = self.path / tag.replace(".", "/")
        cur_p.mkdir(parents=True, exist_ok=True)

        path = cur_p / f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S-%f')}.log"

        if save_type == "json":
            path = path.with_suffix(".json")
            with path.open("w", encoding="utf-8") as f:
                json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)
            return path
        elif save_type == "pkl":
            path = path.with_suffix(".pkl")
            with path.open("wb") as f:
                pickle.dump(obj, f)
            # TODO: save_type: list[Literal["json", "text", "pkl"]]  = ["pkl", "json"]
            # for save_type in save_type:
            try:
                json_path = path.with_suffix(".json")
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return path
        elif save_type == "text":
            obj = str(obj)
            with path.open("w") as f:
                f.write(obj)
            return path

    log_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| "
        r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) *\| "
        r"(?P<caller>.+:.+:\d+) - "
    )

    def iter_msg(self, tag: str | None = None, pattern: str | None = None) -> Generator[Message, None, None]:
        msg_l = []

        if pattern:
            pkl_files = pattern
        elif tag:
            pkl_files = f"**/{tag.replace('.','/')}/**/*.pkl"
        else:
            pkl_files = "**/*.pkl"
        for file in self.path.glob(pkl_files):
            if file.name == "debug_llm.pkl":
                continue
            pkl_log_tag = ".".join(file.relative_to(self.path).as_posix().replace("/", ".").split(".")[:-3])
            pid = file.parent.name

            with file.open("rb") as f:
                content = pickle.load(f)

            timestamp = datetime.strptime(file.stem, "%Y-%m-%d_%H-%M-%S-%f").replace(tzinfo=timezone.utc)

            m = Message(tag=pkl_log_tag, level="INFO", timestamp=timestamp, caller="", pid_trace=pid, content=content)

            msg_l.append(m)

        msg_l.sort(key=lambda x: x.timestamp)
        for m in msg_l:
            yield m

    def truncate(self, time: datetime) -> None:
        for file in self.path.glob("**/*.pkl"):
            timestamp = datetime.strptime(file.stem, "%Y-%m-%d_%H-%M-%S-%f").replace(tzinfo=timezone.utc)
            if timestamp > time.replace(tzinfo=timezone.utc):
                file.unlink()

        _remove_empty_dir(self.path)

    def __str__(self) -> str:
        return f"FileStorage({self.path})"
