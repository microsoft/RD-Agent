from __future__ import annotations

import re
import sys
import pickle
import json
import inspect
from typing import TYPE_CHECKING, Sequence, Literal

if TYPE_CHECKING:
    from loguru import Logger, Message, Record

from loguru import logger
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from functools import partial

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import SingletonBaseClass


def get_caller_info():
    # Get the current stack information
    stack = inspect.stack()
    # The second element is usually the caller's information
    caller_info = stack[2]
    frame = caller_info[0]
    info = {
        'line': caller_info.lineno,
        'name': frame.f_globals['__name__'],  # Get the module name from the frame's globals
    }
    return info


class Storage:
    """
    Basic storage to support saving objects;

    # Usage:

    The storage has mainly two kind of users:
    - The logging end: you can choose any of the following method to use the object
        - We can use it directly with the native logging storage
        - We can use it with other logging tools; For example, serve as a handler for loggers
    - The view end:
        - Mainly for the subclass of `logging.base.View`
        - It should provide two kind of ways to provide content
            - offline content provision.
            - online content preovision.
    """

    @abstractmethod
    def log(self, obj: object, name: str = "", **kwargs: dict) -> str | Path | None:
        """

        Parameters
        ----------
        obj : object
            The object for logging.
        name : str
            The name of the object.  For example "a.b.c"
            We may log a lot of objects to a same name
        """
        ...


class View:
    """
    Motivation:

    Display the content in the storage
    """


class FileStorage(Storage):
    """
    The info are logginged to the file systems

    TODO: describe the storage format
    """

    def __init__(self, path: str = "./log/") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def log(self,
            obj: object,
            name: str = "",
            save_type: Literal["json", "text", "pkl", "short-text"] = "short-text",
            timestamp: datetime | None = None,
            split_name: bool = True,
            ) -> Path:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        
        cur_p = self.path
        if split_name:
            uri_l = name.split(".")
            for u in uri_l:
                cur_p = cur_p / u
        else:
            cur_p = cur_p / name
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
        else:
            obj = str(obj).strip()
            if obj == "":
                return
            path = cur_p / "common_logs.log"
            with path.open("a") as f:
                f.write(f"{timestamp.isoformat()}: {obj}\n\n") # add a new line to separate logs
            return path


class LogColors:
    """
    ANSI color codes for use in console output.
    """

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BLACK = "\033[30m"

    BOLD = "\033[1m"
    ITALIC = "\033[3m"

    END = "\033[0m"

    @classmethod
    def get_all_colors(cls: type[LogColors]) -> list:
        names = dir(cls)
        names = [name for name in names if not name.startswith("__") and not callable(getattr(cls, name))]
        return [getattr(cls, name) for name in names]

    def render(self, text: str, color: str = "", style: str = "") -> str:
        """
        render text by input color and style.
        It's not recommend that input text is already rendered.
        """
        # This method is called too frequently, which is not good.
        colors = self.get_all_colors()
        # Perhaps color and font should be distinguished here.
        if color and color in colors:
            error_message = f"color should be in: {colors} but now is: {color}"
            raise ValueError(error_message)
        if style and style in colors:
            error_message = f"style should be in: {colors} but now is: {style}"
            raise ValueError(error_message)

        text = f"{color}{text}{self.END}"

        return f"{style}{text}{self.END}"


def loguru2storage_handler(storage: Storage, record: Message) -> None:
    msg = f"{record.record['level']} | {record.record['name']}:{record.record['line']} - {RDAgentLog.remove_ansi_codes(record.record['message'])}"
    storage.log(msg, timestamp=record.record["time"], save_type="short-text")


class RDAgentLog(SingletonBaseClass):

    def __init__(self, log_trace_path: str | None = RD_AGENT_SETTINGS.log_trace_path) -> None:
        if log_trace_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f")
            log_trace_path: Path = Path.cwd() / "log" / timestamp
            log_trace_path.mkdir(parents=True, exist_ok=True)
        
        self.storage = FileStorage(log_trace_path)

        # add handler to save log to storage
        logger.add(partial(loguru2storage_handler, self.storage))

        self.log_stream = self.LogStreamContextManager(self.storage)

    @staticmethod
    def remove_ansi_codes(s: str) -> str:
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', s)

    class LogStreamContextManager:
        def __init__(self, storage: Storage) -> None:
            self.captured_logs = []
            self.storage = storage

        def capture(self, message: Message) -> None:
            self.captured_logs.append(message.record["message"])

        def __enter__(self):
            logger.remove()
            logger.add(sys.stderr, format=lambda x: x["message"])
            logger.add(self.capture)

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            logger.info('\n')
            logger.remove()
            logger.add(partial(loguru2storage_handler, self.storage))
            logger.info("[stream log] " + "".join(self.captured_logs))
            logger.add(sys.stderr)

    def log_objects(self, *objs: Sequence[object]) -> None:
        caller_info = get_caller_info()
        for obj in objs:
            logp = self.storage.log(obj, name=f"{type(obj).__module__}.{type(obj).__name__}", save_type="pkl", split_name=False)

            logger.patch(lambda r: r.update(caller_info)).info(f"Logging object in {logp.absolute()}")

    def info(self, msg: str) -> None:
        caller_info = get_caller_info()
        logger.patch(lambda r: r.update(caller_info)).info(msg)

    def warning(self, msg: str) -> None:
        caller_info = get_caller_info()
        logger.patch(lambda r: r.update(caller_info)).warning(msg)

    def error(self, msg: str) -> None:
        caller_info = get_caller_info()
        logger.patch(lambda r: r.update(caller_info)).error(msg)

