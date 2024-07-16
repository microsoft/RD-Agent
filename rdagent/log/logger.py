import sys
import os

from loguru import logger
from rdagent.core.utils import SingletonBaseClass
from rdagent.core.conf import RD_AGENT_SETTINGS
from pathlib import Path
from psutil import Process
from datetime import datetime, timezone
from functools import partial
from contextlib import contextmanager
from multiprocessing import Pipe
from multiprocessing.connection import Connection

from .storage import FileStorage
from .utils import get_caller_info, remove_ansi_codes


class RDAgentLog(SingletonBaseClass):
    _tag: str = ""

    def __init__(self, log_trace_path: str | None = RD_AGENT_SETTINGS.log_trace_path) -> None:
        
        if log_trace_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f")
            log_trace_path: Path = Path.cwd() / "log" / timestamp
        
        self.log_trace_path = Path(log_trace_path)
        self.log_trace_path.mkdir(parents=True, exist_ok=True)
        
        self.storage = FileStorage(log_trace_path)
        
        self.outter_conn, self.inner_conn = Pipe()
        
        self.main_pid = os.getpid()

    @property
    def stream(self) -> Connection:
        return self.outter_conn


    @contextmanager
    def tag(self, tag: str):
        if tag.strip() == "":
            raise ValueError("Tag cannot be empty.")
        if self._tag != "":
            tag = "." + tag

        self._tag = self._tag + tag
        yield
        self._tag = self._tag[:-len(tag)]


    def get_pids(self) -> str:
        '''
        Returns a string of pids from the current process to the main process.
        Split by '-'.
        '''
        pid = os.getpid()
        process = Process(pid)
        pid_chain = f"{pid}"
        while process.pid != self.main_pid:
            parent_pid = process.ppid()
            parent_process = Process(parent_pid)
            pid_chain = f"{parent_pid}-{pid_chain}"
            process = parent_process
        return pid_chain


    def file_format(self, record, raw: bool=False):
        record["message"] = remove_ansi_codes(record["message"])
        if raw:
            return "{message}"
        return "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n"


    def log_object(self, obj: object, *, tag: str = "") -> None:
        caller_info = get_caller_info()
        tag = f"{self._tag}.{tag}".strip('.') + f".{self.get_pids()}"

        logp = self.storage.log(obj, name=tag, save_type="pkl")

        file_handler_id = logger.add(self.log_trace_path / tag.replace('.','/') / "common_logs.log", format=self.file_format)
        logger.patch(lambda r: r.update(caller_info)).info(f"Logging object in {logp.absolute()}")
        logger.remove(file_handler_id)


    def info(self, msg: str, *, tag: str = "", raw: bool=False) -> None:
        caller_info = get_caller_info()
        if raw:
            logger.remove()
            logger.add(sys.stderr, format=lambda r: "{message}")

        tag = f"{self._tag}.{tag}".strip('.') + f".{self.get_pids()}"
        log_file_path = self.log_trace_path / tag.replace('.','/') /"common_logs.log"
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
        caller_info = get_caller_info()

        tag = f"{self._tag}.{tag}".strip('.') + f".{self.get_pids()}"
        file_handler_id = logger.add(self.log_trace_path / tag.replace('.','/') / "common_logs.log", format=self.file_format)
        logger.patch(lambda r: r.update(caller_info)).warning(msg)
        logger.remove(file_handler_id)

    def error(self, msg: str, *, tag: str = "") -> None:
        caller_info = get_caller_info()
        
        tag = f"{self._tag}.{tag}".strip('.') + f".{self.get_pids()}"
        file_handler_id = logger.add(self.log_trace_path / tag.replace('.','/') / "common_logs.log", format=self.file_format)
        logger.patch(lambda r: r.update(caller_info)).error(msg)
        logger.remove(file_handler_id)
