import multiprocessing
import os
import random
import signal
import subprocess
import threading
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
    # if len(returned_msgs):
    #     app.logger.info(data)
    #     app.logger.info([i["tag"] for i in returned_msgs])
    # try:
    #     import json
    #     resp = json.dumps(returned_msgs, ensure_ascii=False)
    # except Exception as e:
    #     app.logger.error(f"Error in jsonify: {e}")
    #     for msg in returned_msgs:
    #         try:
    #             rr = json.dumps(msg, ensure_ascii=False)
    #         except Exception as e:
    #             app.logger.error(f"Error in jsonify individual message: {e}")
    #             app.logger.error(msg)

    return jsonify(returned_msgs), 200


@app.route("/upload", methods=["POST"])
def upload_file():
    # 获取请求体中的字段
    global rdagent_processes, server_port, msgs_for_frontend
    scenario = request.form.get("scenario")
    files = request.files.getlist("files")
    competition = request.form.get("competition")
    loop_n = request.form.get("loops")
    all_duration = request.form.get("all_duration")

    log_folder_path = Path("/home/bowen/workspace/new_traces").absolute()

    if scenario == "Data Science":
        trace_path = log_folder_path / "o1-preview" / f"{competition[10:]}.1"
    else:
        trace_path = log_folder_path / scenario
    id = f"{scenario}/{randomname.get_name()}"

    def read_trace(log_path: Path, t: float = 0.2, id: str = "") -> None:
        from rdagent.log.storage import FileStorage
        from rdagent.log.ui.storage import WebStorage

        fs = FileStorage(log_path)
        ws = WebStorage(port=1, path=log_path)
        msgs_for_frontend[id] = []
        for msg in fs.iter_msg():
            data = ws._obj_to_json(obj=msg.content, tag=msg.tag, id=id, timestamp=msg.timestamp.isoformat())
            if data:
                if isinstance(data, list):
                    for d in data:
                        time.sleep(t)
                        msgs_for_frontend[id].append(d["msg"])
                else:
                    time.sleep(t)
                    msgs_for_frontend[id].append(data["msg"])
        msgs_for_frontend[id].append({"tag": "END", "timestamp": datetime.now(timezone.utc).isoformat(), "content": {}})

    # 启动后台线程，不阻塞 return
    threading.Thread(target=read_trace, args=(trace_path, 0.5, id), daemon=True).start()

    return jsonify({"id": id}), 200


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

    return jsonify({"status": "success", "message": f"Received action '{action}' for process with id '{id}'"})


@app.route("/test", methods=["GET"])
def test():
    # return 'Hello, World!'
    return {k: [i["tag"] for i in v] for k, v in msgs_for_frontend.items()}


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
    app.run(debug=True, host="0.0.0.0", port=port)


if __name__ == "__main__":
    typer.run(main)
