from __future__ import annotations

import os
import random
import re
import sys
import traceback
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from typing import Any

import randomname
from fastapi import UploadFile

from app.config import settings

_TARGETS_WITHOUT_USER_INTERACTION = {"general_model", "fin_factor_report"}


def _secure_filename(filename: str) -> str:
    name = os.path.basename(filename)
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


class RDAgentTask:
    def __init__(
        self,
        target_name: str,
        kwargs: dict,
        stdout_path: str,
        log_trace_path: str,
        scenario: str,
        trace_name: str,
        ui_server_port: int | None = None,
        create_process: bool = True,
    ) -> None:
        self.target_name = target_name
        self.kwargs = kwargs
        self.stdout_path = stdout_path
        self.log_trace_path = log_trace_path
        self.scenario = scenario
        self.trace_name = trace_name
        self.ui_server_port = ui_server_port
        self.process: Process | None = None
        self.user_request_q: Queue = Queue(maxsize=1024)
        self.user_response_q: Queue = Queue(maxsize=1024)

        if create_process:
            self.process = Process(
                target=self._run,
                name=f"rdagent:{self.scenario}:{self.trace_name}",
            )
        self.messages: list[dict] = []
        self.pointers: defaultdict[str, int] = defaultdict(int)

    def start(self) -> None:
        if self.process is not None:
            self.process.start()

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def get_end_code(self) -> int:
        if self.process is None or self.process.exitcode is None:
            return 0
        return self.process.exitcode

    def stop(self) -> None:
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join()
        for q in (self.user_request_q, self.user_response_q):
            try:
                q.cancel_join_thread()
            except Exception:
                pass
            try:
                q.close()
            except Exception:
                pass

    def _run(self) -> None:
        from rdagent.log.conf import LOG_SETTINGS

        LOG_SETTINGS.set_ui_server_port(self.ui_server_port)

        from rdagent.log import rdagent_logger

        rdagent_logger.refresh_storages_from_settings()
        rdagent_logger.set_storages_path(self.log_trace_path)
        Path(self.stdout_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.stdout_path, "w") as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                rdagent_logger.rebind_console_to_current_streams()
                try:
                    if self.target_name not in _TARGETS_WITHOUT_USER_INTERACTION:
                        self.kwargs.setdefault(
                            "user_interaction_queues",
                            (self.user_request_q, self.user_response_q),
                        )
                    if self.target_name == "data_science":
                        from rdagent.app.data_science.loop import main as data_science

                        data_science(**self.kwargs)
                    elif self.target_name == "general_model":
                        from rdagent.app.general_model.general_model import (
                            extract_models_and_implement as general_model,
                        )

                        general_model(**self.kwargs)
                    elif self.target_name == "fin_factor":
                        from rdagent.app.qlib_rd_loop.factor import main as fin_factor

                        fin_factor(**self.kwargs)
                    elif self.target_name == "fin_factor_report":
                        from rdagent.app.qlib_rd_loop.factor_from_report import (
                            main as fin_factor_report,
                        )

                        fin_factor_report(**self.kwargs)
                    elif self.target_name == "fin_model":
                        from rdagent.app.qlib_rd_loop.model import main as fin_model

                        fin_model(**self.kwargs)
                    elif self.target_name == "fin_quant":
                        from rdagent.app.qlib_rd_loop.quant import main as fin_quant

                        fin_quant(**self.kwargs)
                    else:
                        raise ValueError(f"Unknown target: {self.target_name}")
                except Exception:
                    traceback.print_exc()


class AgentRunner:
    SCENARIOS = [
        {
            "name": "Finance Data Building",
            "target": "fin_factor",
            "upload": False,
            "developer": True,
        },
        {
            "name": "Finance Model Implementation",
            "target": "fin_model",
            "upload": False,
            "developer": True,
        },
        {
            "name": "Finance Whole Pipeline",
            "target": "fin_quant",
            "upload": False,
            "developer": True,
        },
        {
            "name": "Finance Data Building (Reports)",
            "target": "fin_factor_report",
            "upload": True,
            "developer": True,
        },
        {
            "name": "General Model Implementation",
            "target": "general_model",
            "upload": True,
            "developer": False,
        },
    ]

    def __init__(self) -> None:
        self.trace_root = settings.trace_folder.resolve()
        self.processes: dict[str, RDAgentTask] = {}

    def list_scenarios(self) -> list[dict[str, Any]]:
        return self.SCENARIOS

    def list_trace_ids(self) -> list[str]:
        self._load_existing_traces()
        return _collect_existing_trace_ids(self.trace_root)

    def get_task(self, relative_trace_id: str) -> RDAgentTask:
        full_id = str(self.trace_root / relative_trace_id)
        return self._get_or_create_task(full_id)

    async def start_run(
        self,
        scenario: str,
        loops: int | None,
        all_duration: str | None,
        files: list[UploadFile],
        competition: str | None = None,
    ) -> str:
        if scenario == "Data Science":
            raise ValueError("Data Science scenario is not supported in terminal UI yet")

        if scenario == "Data Science" and competition:
            trace_name = f"{competition[10:]}-{randomname.get_name()}"
        else:
            trace_name = randomname.get_name()

        trace_files_path = self.trace_root / "uploads" / scenario / trace_name
        log_trace_path = (self.trace_root / scenario / trace_name).absolute()
        stdout_path = self.trace_root / scenario / f"{trace_name}.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)

        for file in files:
            if not file.filename:
                continue
            target_dir = trace_files_path.resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            sanitized = _secure_filename(file.filename)
            target_path = (target_dir / sanitized).resolve()
            if os.path.commonpath([str(target_path), str(target_dir)]) != str(target_dir):
                raise ValueError("Invalid file path")
            content = await file.read()
            target_path.write_bytes(content)

        target_name, kwargs = self._resolve_target(scenario, trace_files_path, loops, all_duration, files)
        task = RDAgentTask(
            target_name=target_name,
            kwargs=kwargs,
            stdout_path=str(stdout_path),
            log_trace_path=str(log_trace_path),
            scenario=scenario,
            trace_name=trace_name,
            ui_server_port=settings.ui_server_port,
        )
        task.start()
        self.processes[str(log_trace_path)] = task
        return f"{scenario}/{trace_name}"

    def ingest_receive_payload(self, payload: dict | list) -> None:
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            trace_id = item.get("id")
            msg = item.get("msg")
            if not trace_id or not msg:
                continue
            task = self._get_or_create_task(trace_id)
            task.messages.append(msg)

    def get_trace_messages(
        self,
        relative_trace_id: str,
        offset: int = 0,
        limit: int = 50,
        return_all: bool = False,
    ) -> list[dict]:
        full_id = str(self.trace_root / relative_trace_id)
        task = self._get_or_create_task(full_id)
        self._drain_user_requests(task)
        self._ensure_end_message(task)

        if return_all or limit <= 0:
            return task.messages[offset:]
        return task.messages[offset : offset + limit]

    def stop_trace(self, relative_trace_id: str) -> None:
        full_id = str(self.trace_root / relative_trace_id)
        task = self.processes.get(full_id)
        if task is None or task.process is None:
            raise KeyError("No running process for given id")
        if task.is_alive():
            task.stop()
        if not task.messages or task.messages[-1].get("tag") != "END":
            task.messages.append(
                {
                    "tag": "END",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "content": {"error_msg": "RD-Agent process was stopped by user.", "end_code": -1},
                }
            )

    def submit_user_interaction(self, relative_trace_id: str, payload: dict) -> None:
        full_id = str(self.trace_root / relative_trace_id)
        task = self._get_or_create_task(full_id)
        task.user_response_q.put(payload, block=False)

    def _resolve_target(
        self,
        scenario: str,
        trace_files_path: Path,
        loops: int | None,
        all_duration: str | None,
        files: list[UploadFile],
    ) -> tuple[str, dict]:
        loop_n_val = loops
        all_duration_val = f"{all_duration}h" if all_duration else None
        kwargs: dict = {}

        if scenario == "Finance Data Building":
            return "fin_factor", {
                "loop_n": loop_n_val,
                "all_duration": all_duration_val,
                "base_features_path": str(trace_files_path),
            }
        if scenario == "Finance Model Implementation":
            return "fin_model", {
                "loop_n": loop_n_val,
                "all_duration": all_duration_val,
                "base_features_path": str(trace_files_path),
            }
        if scenario == "Finance Whole Pipeline":
            return "fin_quant", {
                "loop_n": loop_n_val,
                "all_duration": all_duration_val,
                "base_features_path": str(trace_files_path),
            }
        if scenario == "Finance Data Building (Reports)":
            return "fin_factor_report", {"report_folder": str(trace_files_path), "all_duration": all_duration_val}
        if scenario == "General Model Implementation":
            if files and files[0].filename:
                rfp = str(trace_files_path / _secure_filename(files[0].filename))
            else:
                rfp = str(trace_files_path)
            return "general_model", {"report_file_path": rfp}

        raise ValueError(f"Unknown scenario: {scenario}")

    def _get_or_create_task(self, trace_id: str) -> RDAgentTask:
        task = self.processes.get(trace_id)
        if task is None:
            task = RDAgentTask(
                target_name="",
                kwargs={},
                stdout_path="",
                log_trace_path=trace_id,
                scenario="",
                trace_name="",
                ui_server_port=None,
                create_process=False,
            )
            self.processes[trace_id] = task
        return task

    def _drain_user_requests(self, task: RDAgentTask) -> None:
        try:
            req = task.user_request_q.get_nowait()
        except Empty:
            return
        except Exception:
            return

        if isinstance(req, dict) and {"tag", "timestamp", "content"}.issubset(req.keys()):
            msg = req
        else:
            msg = {
                "tag": "user_interaction.request",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": req,
            }
        task.messages.append(msg)

    def _ensure_end_message(self, task: RDAgentTask) -> None:
        if task.process is not None and not task.is_alive():
            if not task.messages or task.messages[-1].get("tag") != "END":
                task.messages.append(
                    {
                        "tag": "END",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "content": {
                            "error_msg": "RD-Agent process has completed.",
                            "end_code": task.get_end_code(),
                        },
                    }
                )

    def _load_existing_traces(self) -> None:
        for trace_id in _collect_existing_trace_ids(self.trace_root):
            trace_dir = self.trace_root / trace_id
            try:
                read_trace_from_disk(trace_dir, str(trace_dir), self.processes)
            except Exception:
                continue


def _collect_existing_trace_ids(trace_root: Path) -> list[str]:
    if not trace_root.exists():
        return []
    trace_ids: list[str] = []
    for trace_dir in sorted(trace_root.glob("*/*"), key=lambda p: str(p)):
        if not trace_dir.is_dir():
            continue
        if "uploads" in trace_dir.relative_to(trace_root).parts:
            continue
        if not any(trace_dir.rglob("*.pkl")):
            continue
        trace_ids.append(trace_dir.relative_to(trace_root).as_posix())
    return trace_ids


def read_trace_from_disk(log_path: Path, trace_id: str, processes: dict[str, RDAgentTask]) -> None:
    from rdagent.log.storage import FileStorage
    from rdagent.log.ui.storage import WebStorage

    fs = FileStorage(log_path)
    ws = WebStorage(port=1, path=str(log_path))
    task = processes.get(trace_id)
    if task is None:
        task = RDAgentTask(
            target_name="",
            kwargs={},
            stdout_path="",
            log_trace_path=trace_id,
            scenario="",
            trace_name="",
            ui_server_port=None,
            create_process=False,
        )
        processes[trace_id] = task
    task.messages = []
    last_timestamp = None
    for msg in fs.iter_msg():
        data = ws._obj_to_json(obj=msg.content, tag=msg.tag, id=trace_id, timestamp=msg.timestamp.isoformat())
        if data:
            if isinstance(data, list):
                for item in data:
                    task.messages.append(item["msg"])
                last_timestamp = msg.timestamp
            else:
                task.messages.append(data["msg"])
                last_timestamp = msg.timestamp

    now = datetime.now(timezone.utc)
    if last_timestamp and (now - last_timestamp).total_seconds() > 1800:
        task.messages.append(
            {
                "tag": "END",
                "timestamp": now.isoformat(),
                "content": {"error_msg": "Trace session has ended.", "end_code": 0},
            }
        )


agent_runner = AgentRunner()
