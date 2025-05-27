import os
import random
import signal
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

msgs_for_frontend = defaultdict(list)

app = Flask(__name__, static_folder="./docs/_static")
CORS(app)

rdagent_processes = defaultdict()


@app.route("/favicon.ico")
def favicon():
    return send_from_directory("./docs/_static", "favicon.ico", mimetype="image/vnd.microsoft.icon")


pointers = {id: 0 for id in msgs_for_frontend.keys()}


@app.route("/trace", methods=["POST"])
def update_trace():
    data = request.get_json()
    trace_id = data.get("id")
    return_all = data.get("all")
    reset = data.get("reset")
    msg_num = random.randint(1, 10)

    if reset:
        pointers[trace_id] = 0

    end_pointer = pointers[trace_id] + msg_num
    if end_pointer > len(msgs_for_frontend[trace_id]) or return_all:
        end_pointer = len(msgs_for_frontend[trace_id])

    print(f"trace_id: {trace_id}, start_pointer: {pointers[trace_id]}, end_pointer: {end_pointer}")
    returned_msgs = msgs_for_frontend[trace_id][pointers[trace_id] : end_pointer]

    pointers[trace_id] = end_pointer
    return jsonify(returned_msgs), 200


@app.route("/upload", methods=["GET"])
def upload_file():
    # 获取请求体中的字段
    global rdagent_processes
    scenario = request.form.get("scenario")
    files = request.files.getlist("files")
    competition = request.form.get("competition")
    loop_n = request.form.get("loops")
    all_duration = request.form.get("all_duration")


    # scenario = "Data Science Loop"

    # save files
    for file in files:
        if file:
            p = Path(f"./uploads/{scenario}")
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
            file.save(f"./uploads/{scenario}/{file.filename}")

    log_trace_path = Path(f"./RD-Agent_server_trace/{scenario.replace(' ', '_')}/test").absolute()

    if scenario == "Finance Data Building":
        cmds = ["rdagent", "fin_factor"]
        # fin_factor()
        # cmds.append("fin_factor")
    if scenario == "Finance Data Building (Reports)":
        # fin_factor_report(report_folder="git_ignore_folder/reports")
        cmds = ["rdagent", "fin_factor_report", "--report_folder=git_ignore_folder/reports"]
    if scenario == "Finance Model Implementation":
        # fin_model()
        cmds = ["rdagent", "fin_model"]
    if scenario == "General Model Implementation":
        # general_model("https://arxiv.org/pdf/2210.09789")
        cmds = ["rdagent", "general_model", "'https://arxiv.org/pdf/2210.09789'"]
    if scenario == "Medical Model Implementation":
        # med_model()
        cmds = ["rdagent", "med_model"]
    if scenario == "Data Science Loop":
        cmds = ["rdagent", "kaggle", "--competition", competition]

    if loop_n:
        cmds += ["--loop_n", loop_n]
    if all_duration:
        cmds += ["--all_duration", all_duration]

    # subprocess.run(cmds)
    rdagent_processes[str(log_trace_path)] = subprocess.Popen(
        cmds,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env={
            "LOG_TRACE_PATH": str(log_trace_path),
        },
    )

    return (
        jsonify(
            {
                "id": str(log_trace_path),
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
            msgs_for_frontend[d["id"]].append(d["msg"])
    else:
        msgs_for_frontend[data["id"]].append(data["msg"])

    return jsonify({"status": "success"}), 200


@app.route("/control", methods=["POST"])
def control_process():
    global rdagent_processes
    data = request.get_json()
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
            msgs_for_frontend[id].append({"tag": "END", "timestamp": datetime.now(timezone.utc).isoformat(), "content": {}})
            return jsonify({"status": "stopped"}), 200
        else:
            return jsonify({"error": "Unknown action"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to {action} process"}), 500


@app.route("/", methods=["GET"])
def index():
    # return 'Hello, World!'
    return msgs_for_frontend


@app.route("/<path:fn>", methods=["GET"])
def server_static_files(fn):
    return send_from_directory(app.static_folder, fn)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=19899)
