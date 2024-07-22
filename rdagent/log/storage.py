import re
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Generator, Union, Any

from .base import Message, Storage


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
        **kwargs: Any,
    ) -> Union[str, Path]:
        save_type: Literal["json", "text", "pkl"] = kwargs.get("save_type", "text")
        timestamp: Union[datetime, None] = kwargs.get("timestamp", None)

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

    def iter_msg(self, watch: bool = False) -> Generator[Message, None, None]:
        log_pattern = re.compile(
            r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| "
            r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) *\| "
            r"(?P<caller>.+:.+:\d+) - "
        )
        msg_l = []
        for file in self.path.glob("**/*.log"):
            tag = '.'.join(str(file.relative_to(self.path)).replace("/", ".").split(".")[:-3])
            pid = file.parent.name

            with file.open("r") as f:
                content = f.read()

            matches, next_matches = log_pattern.finditer(content), log_pattern.finditer(content)
            next_match = next(next_matches, None)
            # NOTE: the content will be the text between `match` and `next_match`
            for match in matches:
                next_match = next(next_matches, None)

                timestamp_str = match.group("timestamp")
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                level = match.group("level")
                caller = match.group("caller")

                # Extract the message content
                message_start = match.end()
                message_end = next_match.start() if next_match else len(content)
                message_content = content[message_start:message_end].strip()

                m = Message(
                    tag=tag,
                    level=level,
                    timestamp=timestamp,
                    caller=caller,
                    pid_trace=pid,
                    content=message_content
                )

                if isinstance(m.content, str) and "Logging object in" in m.content:
                    absolute_p = m.content.split("Logging object in ")[1]
                    relative_p = "." + absolute_p.split(self.path.name)[1]
                    pkl_path = self.path / relative_p
                    with pkl_path.open("rb") as f:
                        m.content = pickle.load(f)

                msg_l.append(m)

        msg_l.sort(key=lambda x: x.timestamp)
        for m in msg_l:
            yield m
