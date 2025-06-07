import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Literal

from .base import Message, Storage
from .utils import gen_datetime

LOG_LEVEL = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


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

    log_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| "
        r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) *\| "
        r"(?P<caller>.+:.+:\d+) - "
    )

    def iter_msg(self, tag: str | None = None) -> Generator[Message, None, None]:
        msg_l = []

        pkl_files = "**/*.pkl" if tag is None else f"**/{tag.replace('.','/')}/**/*.pkl"
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
            if timestamp > time:
                file.unlink()

        _remove_empty_dir(self.path)

    def __str__(self) -> str:
        return f"FileStorage({self.path})"
