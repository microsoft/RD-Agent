import os
import random
import signal
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import randomname
import typer
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from rdagent.log.ui.conf import UI_SETTING

app = Flask(__name__, static_folder=UI_SETTING.static_path)
CORS(app)

rdagent_processes = defaultdict()
server_port = 19899


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon")


msgs_for_frontend = defaultdict(list)
pointers = defaultdict(int)


def read_trace(log_path: Path, t: float = 0.5) -> None:
    msgs_for_frontend[str(log_path)] = []
    from rdagent.log.storage import FileStorage
    from rdagent.log.ui.storage import WebStorage

    fs = FileStorage(log_path)
    ws = WebStorage(port=1, path=log_path)
    for msg in fs.iter_msg():
        data = ws._obj_to_json(obj=msg.content, tag=msg.tag, id=str(log_path), timestamp=msg.timestamp)
        if data:
            if isinstance(data, list):
                for d in data:
                    time.sleep(t)
                    msgs_for_frontend[str(log_path)].append(d)
            else:
                time.sleep(t)
                msgs_for_frontend[str(log_path)].append(data)


@app.route("/trace", methods=["POST"])
def update_trace():
    global pointers, msgs_for_frontend
    data = request.get_json()
    # app.logger.info(data)
    trace_id = data.get("id")
    return_all = data.get("all")
    reset = data.get("reset")
    msg_num = random.randint(1, 10)

    if reset:
        pointers[trace_id] = 0

    end_pointer = pointers[trace_id] + msg_num
    if end_pointer > len(msgs_for_frontend[trace_id]) or return_all:
        end_pointer = len(msgs_for_frontend[trace_id])

    returned_msgs = msgs_for_frontend[trace_id][pointers[trace_id] : end_pointer]

    pointers[trace_id] = end_pointer
    if len(returned_msgs):
        app.logger.info(data)
        app.logger.info(returned_msgs)
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

    log_folder_path = Path("/home/bowen/workspace/new_traces").absolute()

    if scenario == "Data Science Loop":
        trace_path = log_folder_path / "o1-preview" / f"{competition}.1"
    else:
        trace_path = log_folder_path / scenario

    read_trace(trace_path)

    return jsonify({"id": str(trace_path)}), 200


@app.route("/receive", methods=["POST"])
def receive_msgs():
    try:
        data = request.get_json()
        app.logger.info(data["msg"]["tag"])
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
    global rdagent_processes
    data = request.get_json()
    app.logger.info(data)
    if not data or "id" not in data or "action" not in data:
        return jsonify({"error": "Missing 'id' or 'action' in request"}), 400

    id = data["id"]
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
        return jsonify({"error": f"Failed to {action} process"}), 500


@app.route("/", methods=["GET"])
def index():
    # return 'Hello, World!'
    return str(msgs_for_frontend.keys())


@app.route("/<path:fn>", methods=["GET"])
def server_static_files(fn):
    return send_from_directory(app.static_folder, fn)


def main(port: int = 19899):
    global server_port
    server_port = port
    app.run(debug=True, host="0.0.0.0", port=port)


if __name__ == "__main__":
    typer.run(main)
