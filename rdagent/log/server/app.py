import os
import random
import signal
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import randomname
import typer
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from rdagent.log.storage import FileStorage
from rdagent.log.ui.conf import UI_SETTING
from rdagent.log.ui.storage import WebStorage
from rdagent.log.utils import is_valid_session

app = Flask(__name__, static_folder=UI_SETTING.static_path)
CORS(app)

rdagent_processes = defaultdict()
server_port = 19899
log_folder_path = Path(UI_SETTING.trace_folder).absolute()


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon")


msgs_for_frontend = defaultdict(list)
pointers = defaultdict(lambda: defaultdict(int))  # pointers[trace_id][user_ip]


def read_trace(log_path: Path, id: str = "") -> None:
    fs = FileStorage(log_path)
    ws = WebStorage(port=1, path=log_path)
    msgs_for_frontend[id] = []
    last_timestamp = None
    for msg in fs.iter_msg():
        data = ws._obj_to_json(obj=msg.content, tag=msg.tag, id=id, timestamp=msg.timestamp.isoformat())
        if data:
            if isinstance(data, list):
                for d in data:
                    msgs_for_frontend[id].append(d["msg"])
                    last_timestamp = msg.timestamp
            else:
                msgs_for_frontend[id].append(data["msg"])
                last_timestamp = msg.timestamp

    now = datetime.now(timezone.utc)
    if last_timestamp and (now - last_timestamp).total_seconds() > 1800:
        msgs_for_frontend[id].append({"tag": "END", "timestamp": now.isoformat(), "content": {}})


# load all traces from the log folder
for p in log_folder_path.glob("*/*/"):
    if is_valid_session(p):
        read_trace(p, id=str(p))


@app.route("/trace", methods=["POST"])
def update_trace():
    global pointers, msgs_for_frontend
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

    user_ip = request.remote_addr

    if reset:
        pointers[trace_id][user_ip] = 0

    start_pointer = pointers[trace_id][user_ip]
    end_pointer = start_pointer + msg_num
    if end_pointer > len(msgs_for_frontend[trace_id]) or return_all:
        end_pointer = len(msgs_for_frontend[trace_id])

    returned_msgs = msgs_for_frontend[trace_id][start_pointer:end_pointer]

    pointers[trace_id][user_ip] = end_pointer
    if returned_msgs:
        app.logger.info([msg["tag"] for msg in returned_msgs])
    return jsonify(returned_msgs), 200


@app.route("/upload", methods=["POST"])
def upload_file():
    # 获取请求体中的字段
    global rdagent_processes, server_port
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
    trace_files_path = log_folder_path / scenario / "uploads" / trace_name

    log_trace_path = (log_folder_path / scenario / trace_name).absolute()
    stdout_path = log_folder_path / scenario / f"{trace_name}.stdout"
    if not stdout_path.exists():
        stdout_path.parent.mkdir(parents=True, exist_ok=True)

    # save files
    for file in files:
        if file:
            p = log_folder_path / scenario / "uploads" / trace_name
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
            file.save(p / file.filename)

    if scenario == "Finance Data Building":
        cmds = ["rdagent", "fin_factor"]
    if scenario == "Finance Data Building (Reports)":
        cmds = ["rdagent", "fin_factor_report", "--report_folder", str(trace_files_path)]
    if scenario == "Finance Model Implementation":
        cmds = ["rdagent", "fin_model"]
    if scenario == "General Model Implementation":
        if len(files) == 0:  # files is one link
            rfp = request.form.get("files")[0]
        else:  # one file is uploaded
            rfp = str(trace_files_path / files[0].filename)
        cmds = ["rdagent", "general_model", "--report_file_path", rfp]
    if scenario == "Finance Whole Pipeline":
        cmds = ["rdagent", "fin_quant"]
    if scenario == "Data Science":
        cmds = ["rdagent", "data_science", "--competition", competition]

    # time control parameters
    if scenario != "Finance Data Building (Reports)":
        if loop_n:
            cmds += ["--loop_n", loop_n]
    if all_duration:
        cmds += ["--all_duration", f"{all_duration}h"]

    app.logger.info(f"Started process for {log_trace_path} with parameters: {cmds}")
    with stdout_path.open("w") as log_file:
        rdagent_processes[str(log_trace_path)] = subprocess.Popen(
            cmds,
            stdout=log_file,
            stderr=log_file,
            env={
                **os.environ,
                "LOG_TRACE_PATH": str(log_trace_path),
                "LOG_UI_SERVER_PORT": str(server_port),
            },
        )
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
        # app.logger.info(data["msg"]["tag"])
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
    except Exception as e:
        return jsonify({"error": "Internal Server Error"}), 500

    if isinstance(data, list):
        for d in data:
            msgs_for_frontend[d["id"]].append(d["msg"])
    else:
        msgs_for_frontend[data["id"]].append(data["msg"])

    return jsonify({"status": "success"}), 200


@app.route("/control", methods=["POST"])
def control_process():
    global rdagent_processes, msgs_for_frontend
    data = request.get_json()
    app.logger.info(data)
    if not data or "id" not in data or "action" not in data:
        return jsonify({"error": "Missing 'id' or 'action' in request"}), 400

    id = str(log_folder_path / data["id"])
    action = data["action"]

    if id not in rdagent_processes or rdagent_processes[id] is None:
        return jsonify({"error": "No running process for given id"}), 400

    process = rdagent_processes[id]

    if process.poll() is not None:
        msgs_for_frontend[id].append({"tag": "END", "timestamp": datetime.now(timezone.utc).isoformat(), "content": {}})
        return jsonify({"error": "Process has already terminated"}), 400

    try:
        if action == "pause":
            os.kill(process.pid, signal.SIGSTOP)
            return jsonify({"status": "paused"}), 200
        elif action == "resume":
            os.kill(process.pid, signal.SIGCONT)
            return jsonify({"status": "resumed"}), 200
        elif action == "stop":
            process.terminate()
            process.wait()
            del rdagent_processes[id]
            msgs_for_frontend[id].append(
                {"tag": "END", "timestamp": datetime.now(timezone.utc).isoformat(), "content": {}}
            )
            return jsonify({"status": "stopped"}), 200
        else:
            return jsonify({"error": "Unknown action"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to {action} process, {e}"}), 500


@app.route("/test", methods=["GET"])
def test():
    # return 'Hello, World!'
    global msgs_for_frontend, pointers
    msgs = {k: [i["tag"] for i in v] for k, v in msgs_for_frontend.items()}
    pointers = pointers
    return jsonify({"msgs": msgs, "pointers": pointers}), 200


@app.route("/", methods=["GET"])
def index():
    # return 'Hello, World!'
    # return {k: [i["tag"] for i in v] for k, v in msgs_for_frontend.items()}
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:fn>", methods=["GET"])
def server_static_files(fn):
    return send_from_directory(app.static_folder, fn)


def main(port: int = 19899):
    global server_port
    server_port = port
    app.run(debug=False, host="0.0.0.0", port=port)


if __name__ == "__main__":
    typer.run(main)
