import json
import os
import pickle
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from logging import LogRecord
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, Union

import requests
from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

from psutil import Process

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import SingletonBaseClass

from .base import Storage
from .utils import get_caller_info


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

    def __init__(self, storages: list[Storage] = []) -> None:
        self.storages = storages
        self.main_pid = os.getpid()

    # def set_trace_path(self, log_trace_path: str | Path) -> None:
    #     self.log_trace_path = Path(log_trace_path)
    #     self.storage = FileStorage(log_trace_path)

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

    def log_object(self, obj: object, *, tag: str = "") -> None:
        # TODO: I think we can merge the log_object function with other normal log methods to make the interface simpler.
        caller_info = get_caller_info()
        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")

        for storage in self.storages:
            logp = storage.log(obj, tag=tag)
            logger.patch(lambda r: r.update(caller_info)).info(f"Log object to [{storage}], uri: {logp}")

    def _log(self, level: str, msg: str, *, tag: str = "", raw: bool = False) -> None:
        caller_info = get_caller_info()
        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")

        if raw:
            logger.remove()
            logger.add(sys.stderr, format=lambda r: "{message}")

        log_func = getattr(logger.patch(lambda r: r.update(caller_info)), level)
        log_func(msg)

        if raw:
            logger.remove()
            logger.add(sys.stderr)

    def info(self, msg: str, *, tag: str = "", raw: bool = False) -> None:
        self._log("info", msg, tag=tag, raw=raw)

    def warning(self, msg: str, *, tag: str = "", raw: bool = False) -> None:
        self._log("warning", msg, tag=tag, raw=raw)

    def error(self, msg: str, *, tag: str = "", raw: bool = False) -> None:
        self._log("error", msg, tag=tag, raw=raw)
