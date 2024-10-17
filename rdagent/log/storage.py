import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Literal, Union, cast

from .base import Message, Storage

LOG_LEVEL = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class FileStorage(Storage):
    """
    The info are logginged to the file systems

    TODO: describe the storage format
    """

    def __init__(self, path: str | Path = "./log/") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        obj: object,
        name: str = "",
        save_type: Literal["json", "text", "pkl"] = "text",
        timestamp: datetime | None = None,
        **kwargs: Any,
    ) -> Union[str, Path]:
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

    log_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| "
        r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) *\| "
        r"(?P<caller>.+:.+:\d+) - "
    )

    def iter_msg(self, watch: bool = False) -> Generator[Message, None, None]:
        msg_l = []
        for file in self.path.glob("**/*.log"):
            tag = ".".join(file.relative_to(self.path).as_posix().replace("/", ".").split(".")[:-3])
            pid = file.parent.name

            with file.open("r", encoding="utf-8") as f:
                content = f.read()

            matches, next_matches = self.log_pattern.finditer(content), self.log_pattern.finditer(content)
            next_match = next(next_matches, None)
            # NOTE: the content will be the text between `match` and `next_match`
            for match in matches:
                next_match = next(next_matches, None)

                timestamp_str = match.group("timestamp")
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                level: LOG_LEVEL = cast(LOG_LEVEL, match.group("level"))
                caller = match.group("caller")

                # Extract the message content
                message_start = match.end()
                message_end = next_match.start() if next_match else len(content)
                message_content = content[message_start:message_end].strip()

                if "Logging object in" in message_content:
                    continue

                m = Message(
                    tag=tag, level=level, timestamp=timestamp, caller=caller, pid_trace=pid, content=message_content
                )

                msg_l.append(m)

        for file in self.path.glob("**/*.pkl"):
            tag = ".".join(file.relative_to(self.path).as_posix().replace("/", ".").split(".")[:-3])
            pid = file.parent.name

            with file.open("rb") as f:
                content = pickle.load(f)

            timestamp = datetime.strptime(file.stem, "%Y-%m-%d_%H-%M-%S-%f").replace(tzinfo=timezone.utc)

            m = Message(tag=tag, level="INFO", timestamp=timestamp, caller="", pid_trace=pid, content=content)

            msg_l.append(m)

        msg_l.sort(key=lambda x: x.timestamp)
        for m in msg_l:
            yield m

    def truncate(self, time: datetime) -> None:
        # any message later than `time` will be removed
        for file in self.path.glob("**/*.log"):
            with file.open("r") as f:
                content = f.read()

            new_content = ""

            matches, next_matches = self.log_pattern.finditer(content), self.log_pattern.finditer(content)

            next_match = next(next_matches, None)
            for match in matches:
                next_match = next(next_matches, None)
                timestamp_str = match.group("timestamp")
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)

                log_start = match.start()
                log_end = next_match.start() if next_match else len(content)
                msg = content[match.end() : log_end].strip()

                if timestamp > time:
                    if "Logging object in" in msg:
                        absolute_p = msg.split("Logging object in ")[1]
                        p = Path(absolute_p)
                        p.unlink()
                    continue

                new_content += content[log_start:log_end]
            with file.open("w") as f:
                f.write(new_content)
