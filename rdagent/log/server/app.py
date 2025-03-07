import os
import random
import signal
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# %%
msgs_for_frontend = defaultdict(list)

app = Flask(__name__, static_folder="./docs/_static")
CORS(app)

# %%
base_path = Path("./log")
dir2id = {dir_name.name: idx for idx, dir_name in enumerate(base_path.iterdir())}

fin_factor_report_proc = None

"""
for dn, did in dir2id.items():
    fs = FileStorage(base_path / dn)
    for m in fs.iter_msg():
        if 'r.hypothesis generation' in m.tag:
            h: Hypothesis = m.content
            msgs_for_frontend[did].append({
                "tag": "research.hypothesis",
                "timestamp": m.timestamp.isoformat(),
                "content": {
                    "name_map": {
                        "hypothesis": "RD-Agent proposes the hypothesis⬇️",
                        "concise_justification": "because the reason⬇️",
                        "concise_observation": "based on the observation⬇️",
                        "concise_knowledge": "Knowledge⬇️ gained after practice",
                    },
                    "hypothesis": h.hypothesis,
                    "reason": h.reason,
                    "concise_reason": h.concise_reason,
                    "concise_justification": h.concise_justification,
                    "concise_observation": h.concise_observation,
                    "concise_knowledge": h.concise_knowledge,
                }
            })
        elif 'r.pdf_image' in m.tag:
            m.content.save(f"{app.static_folder}/{m.timestamp.timestamp()}.jpg")
            msgs_for_frontend[did].append({
                "tag": "research.pdf_image",
                "timestamp": m.timestamp.isoformat(),
                "content": {
                    "image": f"{m.timestamp.timestamp()}.jpg"
                }
            })
        elif 'load_pdf_screenshot' in m.tag:
            m.content.save(f"{app.static_folder}/{m.timestamp.timestamp()}.jpg")
            msgs_for_frontend[did].append({
                "tag": "research.pdf_image",
                "timestamp": m.timestamp.isoformat(),
                "content": {
                    "image": f"{m.timestamp.timestamp()}.jpg"
                }
            })
        elif 'r.experiment generation' in m.tag or 'd.load_experiment' in m.tag:
            if 'd.load_experiment' in m.tag:
                tasks: list[FactorTask | ModelTask] = m.content.sub_tasks
            else:
                tasks: list[FactorTask | ModelTask] = m.content
            if isinstance(tasks[0], FactorTask):
                msgs_for_frontend[did].append({
                    "tag": "research.tasks",
                    "timestamp": m.timestamp.isoformat(),
                    "content": [
                        {
                            "name": t.factor_name,
                            "description": t.factor_description,
                            "formulation": t.factor_formulation,
                            "variables": t.variables,
                        } for t in tasks
                    ]
                })
            elif isinstance(tasks[0], ModelTask):
                msgs_for_frontend[did].append({
                    "tag": "research.tasks",
                    "timestamp": m.timestamp.isoformat(),
                    "content": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "model_type": t.model_type,
                            "formulation": t.formulation,
                            "variables": t.variables,
                        } for t in tasks
                    ]
                })
        elif "d.evolving code" in m.tag:
            ws: list[FactorFBWorkspace | ModelFBWorkspace] = [i for i in m.content]
            msgs_for_frontend[did].append({
                "tag": "evolving.codes",
                "timestamp": m.timestamp.isoformat(),
                "content": [
                    {
                        "target_task_name": w.target_task.name if isinstance(w.target_task, ModelTask) else w.target_task.factor_name,
                        "code": w.code_dict
                    } for w in ws if w
                ]
            })
        elif "d.evolving feedback" in m.tag:
            fl: list[FactorSingleFeedback | ModelCoderFeedback] = [i for i in m.content]
            msgs_for_frontend[did].append({
                "tag": "evolving.feedbacks",
                "timestamp": m.timestamp.isoformat(),
                "content": [
                    {
                        "final_decision": f.final_decision,
                        "final_feedback": f.final_feedback,
                        "execution_feedback": f.execution_feedback,
                        "code_feedback": f.code_feedback,
                        "value_feedback": f.value_feedback if isinstance(f, ModelCoderFeedback) else f.factor_value_feedback,
                        "model_shape_feedback": f.shape_feedback if isinstance(f, ModelCoderFeedback) else None,
                    } for f in fl if f
                ]
            })
        elif "scenario" in m.tag:
            msgs_for_frontend[did].append({
                "tag": "feedback.config",
                "timestamp": m.timestamp.isoformat(),
                "content": {
                    "config": m.content.experiment_setting
                }
            })
        elif "ef.Quantitative Backtesting Chart" in m.tag:
            msgs_for_frontend[did].append({
                "tag": "feedback.return_chart",
                "timestamp": m.timestamp.isoformat(),
                "content": {
                    "chart_html": plotly.io.to_html(report_figure(m.content))
                }
            })
        elif "model runner result" in m.tag or "factor runner result" in m.tag or "runner result" in m.tag:
            msgs_for_frontend[did].append({
                "tag": "feedback.metric",
                "timestamp": m.timestamp.isoformat(),
                "content": {
                    "result": m.content.result.to_json(),
                }
            })
        elif "ef.feedback" in m.tag:
            hf: HypothesisFeedback = m.content
            msgs_for_frontend[did].append({
                "tag": "feedback.hypothesis_feedback",
                "timestamp": m.timestamp.isoformat(),
                "content": {
                    "observations": hf.observations,
                    "hypothesis_evaluation": hf.hypothesis_evaluation,
                    "new_hypothesis": hf.new_hypothesis,
                    "decision": hf.decision,
                    "reason": hf.reason,
                }
            })

# add END message
for msgs in msgs_for_frontend.values():
    msgs.append({
        "tag": "END",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content": {}
    })
"""


@app.route("/favicon.ico")
def favicon():
    return send_from_directory("./docs/_static", "favicon.ico", mimetype="image/vnd.microsoft.icon")


# %%
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
    return jsonify(returned_msgs) if len(returned_msgs) > 0 else jsonify([{"tag": "END", "content": {}}]), 200


@app.route("/upload", methods=["GET"])
def upload_file():
    # 获取请求体中的字段
    global fin_factor_report_proc
    if fin_factor_report_proc is not None:
        return jsonify({"error": "fin_factor_report is already running"}), 400
    # scenario = request.form.get('scenario')
    # files = request.files.getlist('files')
    # competition = request.form.get('competition')

    scenario = "Data Science Loop"

    # 检查必要字段
    if not scenario:
        return jsonify({"id": -1, "success": False}), 400

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
        cmds = ["rdagent", "ds_loop", "--competition", "aerial-cactus-identification"]

    # subprocess.run(cmds)
    fin_factor_report_proc = subprocess.Popen(
        cmds,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    return jsonify({"status": "success"}), 200

    # if scenario == 'kaggle' and not competition:
    #     return jsonify({'error': 'competition is required for kaggle scenario'}), 400

    # 处理文件上传
    # for file in files:
    #     if file:
    #         p = Path(f'./uploads/{scenario}')
    #         if not p.exists():
    #             p.mkdir(parents=True, exist_ok=True)
    #         file.save(f'./uploads/{scenario}/{file.filename}')

    # id = dir2id[scenario]

    # return jsonify({
    #     'id': id,
    # }), 200


@app.route("/receive", methods=["POST"])
def receive_msgs():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
    except Exception:
        return jsonify({"error": "Internal Server Error"}), 500

    msgs_for_frontend[data["id"]].append(data["msg"])

    print(msgs_for_frontend)
    return jsonify({"status": "success", "received": data}), 200


@app.route("/pause", methods=["GET"])
def pause_process():
    global fin_factor_report_proc
    if fin_factor_report_proc is None:
        return jsonify({"error": "No running process to pause"}), 400

    if fin_factor_report_proc.poll() is not None:
        return jsonify({"error": "Process is not running"}), 400

    try:
        os.kill(fin_factor_report_proc.pid, signal.SIGSTOP)
        return jsonify({"status": "paused"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/resume", methods=["GET"])
def resume_process():
    global fin_factor_report_proc
    if fin_factor_report_proc is None:
        return jsonify({"error": "No running process to pause"}), 400

    if fin_factor_report_proc.poll() is not None:
        return jsonify({"error": "Process is not running"}), 400

    try:
        os.kill(fin_factor_report_proc.pid, signal.SIGCONT)
        return jsonify({"status": "paused"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stop", methods=["GET"])
def stop_process():
    global fin_factor_report_proc
    if fin_factor_report_proc is None:
        return jsonify({"error": "No running process to pause"}), 400

    if fin_factor_report_proc.poll() is not None:
        return jsonify({"error": "Process is not running"}), 400

    try:
        fin_factor_report_proc.terminate()
        fin_factor_report_proc.wait()
        fin_factor_report_proc = None
        return jsonify({"status": "stopped"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    # return 'Hello, World!'
    return msgs_for_frontend


@app.route("/<path:fn>", methods=["GET"])
def server_static_files(fn):
    return send_from_directory(app.static_folder, fn)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=19899)
