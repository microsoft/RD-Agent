import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Generator

from loguru import logger

from .conf import LOG_SETTINGS

from psutil import Process

from rdagent.core.utils import SingletonBaseClass, import_class

from .base import Storage
from .storage import FileStorage
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

    # Thread-/coroutine-local tag;  In Linux forked subprocess, it will be copied to the subprocess.
    _tag_ctx: ContextVar[str] = ContextVar("_tag_ctx", default="")
    _raw_log_key = "_rdagent_raw"

    @classmethod
    def _configure_console_sinks(cls) -> None:
        raw_filter = lambda record: bool(record["extra"].get(cls._raw_log_key, False))
        normal_filter = lambda record: not raw_filter(record)

        if LOG_SETTINGS.format_console is not None:
            logger.add(sys.stdout, format=LOG_SETTINGS.format_console, filter=normal_filter)
        else:
            logger.add(sys.stdout, filter=normal_filter)
        logger.add(sys.stdout, format="{message}", filter=raw_filter)

    @property
    def _tag(self) -> str:  # Get current tag
        return self._tag_ctx.get()

    @_tag.setter  # Set current tag
    def _tag(self, value: str) -> None:
        self._tag_ctx.set(value)

    def __init__(self) -> None:
        logger.remove()
        self._configure_console_sinks()

        self.storage = FileStorage(LOG_SETTINGS.trace_path)
        self.other_storages: list[Storage] = []
        self.refresh_storages_from_settings()

        self.main_pid = os.getpid()

    def refresh_storages_from_settings(self) -> None:
        self.other_storages = []
        for storage, args in LOG_SETTINGS.storages.items():
            storage_cls = import_class(storage)
            self.other_storages.append(storage_cls(*args))

    def rebind_console_to_current_streams(self) -> None:
        """Rebind loguru sinks to the current stdio objects.

        This is needed in forked/spawned subprocesses after stdout/stderr have been
        redirected, because loguru keeps references to the original stream objects.
        """
        logger.remove()
        self._configure_console_sinks()

    @contextmanager
    def tag(self, tag: str) -> Generator[None, None, None]:
        if tag.strip() == "":
            raise ValueError("Tag cannot be empty.")
        # Generate a new complete tag
        current_tag = self._tag_ctx.get()
        new_tag = tag if current_tag == "" else f"{current_tag}.{tag}"
        # Set and save token for later restore
        token = self._tag_ctx.set(new_tag)
        try:
            yield
        finally:
            # Restore previous tag (thread/coroutine safe)
            self._tag_ctx.reset(token)

    def set_storages_path(self, path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)
        for storage in [self.storage] + self.other_storages:
            if hasattr(storage, "path"):
                storage.path = path

    def truncate_storages(self, time: datetime) -> None:
        for storage in [self.storage] + self.other_storages:
            storage.truncate(time=time)

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
        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")

        for storage in [self.storage] + self.other_storages:
            storage.log(obj, tag=tag)

    def _log(self, level: str, msg: str, *, tag: str = "", raw: bool = False) -> None:
        caller_info = get_caller_info(level=3)
        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")

        patched_logger = logger.patch(lambda r: r.update(caller_info)).bind(**{self._raw_log_key: raw}).opt(raw=raw)
        log_func = getattr(patched_logger, level)
        log_func(msg)

    def info(self, msg: str, *, tag: str = "", raw: bool = False) -> None:
        self._log("info", msg, tag=tag, raw=raw)

    def warning(self, msg: str, *, tag: str = "", raw: bool = False) -> None:
        self._log("warning", msg, tag=tag, raw=raw)

    def error(self, msg: str, *, tag: str = "", raw: bool = False) -> None:
        self._log("error", msg, tag=tag, raw=raw)
