import logging
import os
import random
import traceback
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty

import randomname
import typer
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from rdagent.log.storage import FileStorage
from rdagent.log.ui.conf import UI_SETTING
from rdagent.log.ui.storage import WebStorage

app = Flask(__name__, static_folder=str(Path(UI_SETTING.static_path).resolve()))
CORS(app)
app.config["UI_SERVER_PORT"] = 19899

_YELLOW = "\033[33m"
_RESET = "\033[0m"


class _YellowWarningFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.WARNING:
            record.levelname = f"{_YELLOW}{record.levelname}{_RESET}"
        return super().format(record)


def _configure_app_logger() -> None:
    formatter = _YellowWarningFormatter(
        fmt="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in app.logger.handlers:
        handler.setFormatter(formatter)


_configure_app_logger()


_TARGETS_WITHOUT_USER_INTERACTION = {"general_model", "fin_factor_report"}


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

        # Two IPC queues for user interaction.
        # - `user_request_q`: rdagent subprocess -> server (dicts to render on frontend)
        # - `user_response_q`: server -> rdagent subprocess (user input dicts)
        # NOTE: Use multiprocessing.Queue because rdagent is started as a separate process.
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

        # Best-effort cleanup for IPC queues.
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
                    # Only interactive targets should receive IPC queues.
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


rdagent_processes: dict[str, RDAgentTask] = {}
log_folder_path = Path(UI_SETTING.trace_folder).absolute()


def _drain_user_requests_into_messages(task: RDAgentTask) -> None:
    """Move a single pending user-interaction request into `task.messages`.

    Assumption: each rdagent process only has one active request at a time.
    """

    try:
        req = task.user_request_q.get_nowait()
    except Empty:
        return
    except Exception:
        return

    # Standardize the message shape for the frontend.
    # The agent can send either a full message dict, or a raw content dict.
    if isinstance(req, dict) and {"tag", "timestamp", "content"}.issubset(req.keys()):
        msg = req
    else:
        msg = {
            "tag": "user_interaction.request",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": req,
        }
    task.messages.append(msg)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon")


def _normalize_static_request_path(fn: str) -> str:
    static_prefix = UI_SETTING.static_path.strip("./")
    if static_prefix and fn.startswith(f"{static_prefix}/"):
        return fn[len(static_prefix) + 1 :]
    return fn


def _get_or_create_task(trace_id: str) -> RDAgentTask:
    task = rdagent_processes.get(trace_id)
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
        rdagent_processes[trace_id] = task
    return task


def _resolve_stdout_path(trace_id: str) -> Path | None:
    normalized_trace_id = str(trace_id or "").strip()
    if not normalized_trace_id:
        return None

    task = rdagent_processes.get(str(log_folder_path / normalized_trace_id))
    if task is None or not task.stdout_path:
        return None

    stdout_path = Path(task.stdout_path).resolve()

    try:
        if os.path.commonpath([str(stdout_path), str(log_folder_path)]) != str(log_folder_path):
            return None
    except ValueError:
        return None

    return stdout_path


def read_trace(log_path: Path, id: str = "") -> None:
    fs = FileStorage(log_path)
    ws = WebStorage(port=1, path=log_path)
    task = _get_or_create_task(id)
    task.messages = []
    last_timestamp = None
    for msg in fs.iter_msg():
        data = ws._obj_to_json(obj=msg.content, tag=msg.tag, id=id, timestamp=msg.timestamp.isoformat())
        if data:
            if isinstance(data, list):
                for d in data:
                    task.messages.append(d["msg"])
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


# load all traces from the log folder
# for p in log_folder_path.glob("*/*/"):
#     read_trace(p, id=str(p))


@app.route("/trace", methods=["POST"])
def update_trace():
    data = request.get_json()
    trace_id = data.get("id")
    return_all = data.get("all")
    reset = data.get("reset")
    msg_num = random.randint(1, 10)
    app.logger.info(data)
    log_folder_path = Path(UI_SETTING.trace_folder).absolute()
    if not trace_id:
        return jsonify({"error": "Trace ID is required"}), 400
    trace_id = str(log_folder_path / trace_id)

    task = _get_or_create_task(trace_id)

    # Make sure any pending user-interaction requests are visible to the frontend.
    _drain_user_requests_into_messages(task)

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
            app.logger.warning(f"Process for {trace_id} has ended.")

    user_ip = request.remote_addr

    if reset:
        task.pointers[user_ip] = 0

    start_pointer = task.pointers[user_ip]
    end_pointer = start_pointer + msg_num
    if end_pointer > len(task.messages) or return_all:
        end_pointer = len(task.messages)

    returned_msgs = task.messages[start_pointer:end_pointer]
    task.pointers[user_ip] = end_pointer
    if returned_msgs:
        app.logger.info([msg["tag"] for msg in returned_msgs])
    return jsonify(returned_msgs), 200


@app.route("/stdout", methods=["GET"])
def download_stdout_file():
    trace_id = request.args.get("id", "")
    stdout_path = _resolve_stdout_path(trace_id)

    if stdout_path is None:
        return jsonify({"error": "Trace ID is required or invalid"}), 400
    if not stdout_path.exists() or not stdout_path.is_file():
        return jsonify({"error": "Stdout file not found"}), 404

    return send_file(
        stdout_path,
        as_attachment=True,
        download_name=stdout_path.name,
        mimetype="text/plain",
    )


@app.route("/upload", methods=["POST"])
def upload_file():
    # 获取请求体中的字段
    global rdagent_processes
    scenario = request.form.get("scenario")
    files = request.files.getlist("files")
    competition = request.form.get("competition")
    loop_n = request.form.get("loops")
    all_duration = request.form.get("all_duration")

    # scenario = "Data Science Loop"
    if scenario == "Data Science":
        competition = competition[10:]  # Eg. MLE-Bench:aerial-cactus-competition
        trace_name = f"{competition}-{randomname.get_name()}"
    else:
        trace_name = randomname.get_name()
    trace_files_path = log_folder_path / "uploads" / scenario / trace_name

    log_trace_path = (log_folder_path / scenario / trace_name).absolute()
    stdout_path = log_folder_path / scenario / f"{trace_name}.log"
    if not stdout_path.exists():
        stdout_path.parent.mkdir(parents=True, exist_ok=True)

    # save files
    for file in files:
        if file:
            p = (log_folder_path / "uploads" / scenario / trace_name).resolve()
            sanitized_filename = secure_filename(file.filename)  # Sanitize filename
            target_path = (p / sanitized_filename).resolve()  # Normalize target path
            # Ensure target_path is within the allowed base directory
            if os.path.commonpath([str(target_path), str(p)]) == str(p) and target_path.is_file() == False:
                if not p.exists():
                    p.mkdir(parents=True, exist_ok=True)
                file.save(target_path)
            else:
                return jsonify({"error": "Invalid file path"}), 400

    target_name = None
    kwargs = {}
    loop_n_val = int(loop_n) if loop_n else None
    all_duration_val = f"{all_duration}h" if all_duration else None

    if scenario == "Finance Data Building":
        target_name = "fin_factor"
        kwargs = {
            "loop_n": loop_n_val,
            "all_duration": all_duration_val,
            "base_features_path": str(trace_files_path),
        }
    if scenario == "Finance Model Implementation":
        target_name = "fin_model"
        kwargs = {
            "loop_n": loop_n_val,
            "all_duration": all_duration_val,
            "base_features_path": str(trace_files_path),
        }
    if scenario == "Finance Whole Pipeline":
        target_name = "fin_quant"
        kwargs = {
            "loop_n": loop_n_val,
            "all_duration": all_duration_val,
            "base_features_path": str(trace_files_path),
        }
    if scenario == "Finance Data Building (Reports)":
        target_name = "fin_factor_report"
        kwargs = {"report_folder": str(trace_files_path), "all_duration": all_duration_val}
    if scenario == "General Model Implementation":
        if len(files) == 0:  # files is one link
            rfp = request.form.get("files")[0]
        else:  # one file is uploaded
            rfp = str(trace_files_path / files[0].filename)
        target_name = "general_model"
        kwargs = {"report_file_path": rfp}
    if scenario == "Data Science":
        target_name = "data_science"
        kwargs = {"competition": competition, "loop_n": loop_n_val, "timeout": all_duration_val}

    if target_name is None:
        return jsonify({"error": "Unknown scenario"}), 400

    app.logger.info(f"Started process for {log_trace_path} with target: {target_name}, kwargs: {kwargs}")
    task = RDAgentTask(
        target_name=target_name,
        kwargs=kwargs,
        stdout_path=str(stdout_path),
        log_trace_path=str(log_trace_path),
        scenario=scenario,
        trace_name=trace_name,
        ui_server_port=app.config["UI_SERVER_PORT"],
    )
    task.start()
    app.logger.warning(f"Task {log_trace_path} started.")
    rdagent_processes[str(log_trace_path)] = task
    return (
        jsonify(
            {
                "id": f"{scenario}/{trace_name}",
            }
        ),
        200,
    )


@app.route("/receive", methods=["POST"])
def receive_msgs():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
    except Exception as e:
        return jsonify({"error": "Internal Server Error"}), 500

    if isinstance(data, list):
        for d in data:
            task = _get_or_create_task(d["id"])
            task.messages.append(d["msg"])
    else:
        task = _get_or_create_task(data["id"])
        task.messages.append(data["msg"])

    return jsonify({"status": "success"}), 200


@app.route("/user_interaction/submit", methods=["POST"])
def submit_user_interaction_response():
    """Frontend submits a user response; server forwards it to the rdagent subprocess via IPC queue."""
    data = request.get_json(silent=True) or {}
    trace_id = data.get("id")
    payload = data.get("payload")

    if not trace_id:
        return jsonify({"error": "Trace ID is required"}), 400
    if payload is None:
        return jsonify({"error": "Missing 'payload'"}), 400

    trace_id = str(log_folder_path / trace_id)
    task = _get_or_create_task(trace_id)

    try:
        task.user_response_q.put(payload, block=False)
    except Exception as e:
        return jsonify({"error": f"Failed to enqueue user response: {e}"}), 500

    return jsonify({"status": "success"}), 200


@app.route("/control", methods=["POST"])
def control_process():
    global rdagent_processes
    data = request.get_json()
    app.logger.info(data)
    if not data or "id" not in data or "action" not in data:
        return jsonify({"error": "Missing 'id' or 'action' in request"}), 400

    id = str(log_folder_path / data["id"])
    action = data["action"]

    if action != "stop":
        return jsonify({"error": "Only 'stop' action is supported"}), 400

    if id not in rdagent_processes or rdagent_processes[id] is None:
        return jsonify({"error": "No running process for given id"}), 400

    task = rdagent_processes[id]

    if task.process is None:
        return jsonify({"error": "No running process for given id"}), 400

    try:
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
            app.logger.warning(f"Process for {id} has been stopped.")
        return jsonify({"status": "stopped"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to {action} process, {e}"}), 500


@app.route("/test", methods=["GET"])
def test():
    # return 'Hello, World!'
    msgs = {k: [i["tag"] for i in task.messages] for k, task in rdagent_processes.items()}
    pointers = {k: dict(task.pointers) for k, task in rdagent_processes.items()}
    return jsonify({"msgs": msgs, "pointers": pointers}), 200


@app.route("/", methods=["GET"])
def index():
    # return 'Hello, World!'
    # return {k: [i["tag"] for i in v] for k, v in msgs_for_frontend.items()}
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:fn>", methods=["GET"])
def server_static_files(fn):
    return send_from_directory(app.static_folder, _normalize_static_request_path(fn))


def main(port: int = 19899):
    app.config["UI_SERVER_PORT"] = port
    app.run(debug=False, host="0.0.0.0", port=port)


if __name__ == "__main__":
    typer.run(main)
