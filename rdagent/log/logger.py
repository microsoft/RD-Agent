import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from logging import LogRecord
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, Union

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

from psutil import Process

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import SingletonBaseClass

from .storage import FileStorage
from .utils import LogColors, get_caller_info


class RDAgentLog(SingletonBaseClass):
    """
    The files are organized based on the tag & PID
    Here is an example tag

    .. code-block::

        a
        - b
        - c
            - 123
              - common_logs.log
            - 1322
              - common_logs.log
            - 1233
              - <timestamp>.pkl
            - d
                - 1233-673 ...
                - 1233-4563 ...
                - 1233-365 ...

    """

    # TODO: Simplify it to introduce less concepts ( We may merge RDAgentLog, Storage &)
    # Solution:  Storage => PipeLog, View => PipeLogView, RDAgentLog is an instance of PipeLogger
    # PipeLogger.info(...) ,  PipeLogger.get_resp() to get feedback from frontend.
    # def f():
    #   logger = PipeLog()
    #   logger.info("<code>")
    #   feedback = logger.get_reps()
    _tag: str = ""

    def __init__(self, log_trace_path: Union[str, None] = RD_AGENT_SETTINGS.log_trace_path) -> None:
        if log_trace_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f")
            self.log_trace_path = Path.cwd() / "log" / timestamp
        else:
            self.log_trace_path = Path(log_trace_path)

        self.log_trace_path.mkdir(parents=True, exist_ok=True)

        self.storage = FileStorage(self.log_trace_path)

        self.main_pid = os.getpid()

    def set_trace_path(self, log_trace_path: str | Path) -> None:
        self.log_trace_path = Path(log_trace_path)
        self.storage = FileStorage(log_trace_path)

    @contextmanager
    def tag(self, tag: str) -> Generator[None, None, None]:
        if tag.strip() == "":
            raise ValueError("Tag cannot be empty.")
        if self._tag != "":
            tag = "." + tag

        # TODO: It may result in error in mutithreading or co-routine
        self._tag = self._tag + tag
        try:
            yield
        finally:
            self._tag = self._tag[: -len(tag)]

    def get_pids(self) -> str:
        """
        Returns a string of pids from the current process to the main process.
        Split by '-'.
        """
        pid = os.getpid()
        process = Process(pid)
        pid_chain = f"{pid}"
        while process.pid != self.main_pid:
            parent_pid = process.ppid()
            parent_process = Process(parent_pid)
            pid_chain = f"{parent_pid}-{pid_chain}"
            process = parent_process
        return pid_chain

    def file_format(self, record: "Record", raw: bool = False) -> str:
        # FIXME: the formmat is tightly coupled with the message reading in storage.
        record["message"] = LogColors.remove_ansi_codes(record["message"])
        if raw:
            return "{message}"
        return "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n"

    def log_object(self, obj: object, *, tag: str = "") -> None:
        # TODO: I think we can merge the log_object function with other normal log methods to make the interface simpler.
        caller_info = get_caller_info()
        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")

        logp = self.storage.log(obj, name=tag, save_type="pkl")

        file_handler_id = logger.add(
            self.log_trace_path / tag.replace(".", "/") / "common_logs.log", format=self.file_format
        )
        logger.patch(lambda r: r.update(caller_info)).info(f"Logging object in {Path(logp).absolute()}")
        logger.remove(file_handler_id)

    def info(self, msg: str, *, tag: str = "", raw: bool = False) -> None:
        # TODO: too much duplicated. due to we have no logger with stream context;
        caller_info = get_caller_info()
        if raw:
            logger.remove()
            logger.add(sys.stderr, format=lambda r: "{message}")

        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")
        log_file_path = self.log_trace_path / tag.replace(".", "/") / "common_logs.log"
        if raw:
            file_handler_id = logger.add(log_file_path, format=partial(self.file_format, raw=True))
        else:
            file_handler_id = logger.add(log_file_path, format=self.file_format)

        logger.patch(lambda r: r.update(caller_info)).info(msg)
        logger.remove(file_handler_id)

        if raw:
            logger.remove()
            logger.add(sys.stderr)

    def warning(self, msg: str, *, tag: str = "") -> None:
        # TODO: reuse code
        # _log(self, msg: str, *, tag: str = "", level=Literal["warning", "error", ..]) -> None:
        # getattr(logger.patch(lambda r: r.update(caller_info)), level)(msg)
        caller_info = get_caller_info()

        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")
        file_handler_id = logger.add(
            self.log_trace_path / tag.replace(".", "/") / "common_logs.log", format=self.file_format
        )
        logger.patch(lambda r: r.update(caller_info)).warning(msg)
        logger.remove(file_handler_id)

    def error(self, msg: str, *, tag: str = "") -> None:
        caller_info = get_caller_info()

        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")
        file_handler_id = logger.add(
            self.log_trace_path / tag.replace(".", "/") / "common_logs.log", format=self.file_format
        )
        logger.patch(lambda r: r.update(caller_info)).error(msg)
        logger.remove(file_handler_id)
