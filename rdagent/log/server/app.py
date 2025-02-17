from collections import defaultdict
from pathlib import Path
import random
from rdagent.log.storage import Message, FileStorage
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import plotly

from rdagent.components.coder.factor_coder.CoSTEER.evaluators import FactorSingleFeedback
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.components.coder.model_coder.CoSTEER.evaluators import ModelCoderFeedback
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.log.storage import FileStorage
from rdagent.log.ui.qlib_report_figure import report_figure



app = Flask(__name__, static_folder='static')
CORS(app)

#%%
base_path = Path('./demo_traces')
dir2id = {dir_name.name: idx for idx, dir_name in enumerate(base_path.iterdir())}
#%%
msgs_for_frontend = defaultdict(list)

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

#%%
pointers = {
    id: 0 for id in msgs_for_frontend.keys()
}

@app.route('/trace', methods=['POST'])
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
    returned_msgs = msgs_for_frontend[trace_id][pointers[trace_id]:end_pointer]
    
    pointers[trace_id] = end_pointer
    return jsonify(returned_msgs) if len(returned_msgs) > 0 else jsonify([{ "tag": "END", "content": {} }]), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    # 获取请求体中的字段
    scenario = request.form.get('scenario')
    files = request.files.getlist('files')
    competition = request.form.get('competition')

    # 检查必要字段
    if not scenario:
        return jsonify({'id': -1, "success": False}), 400

    # if scenario == 'kaggle' and not competition:
    #     return jsonify({'error': 'competition is required for kaggle scenario'}), 400

    # 处理文件上传
    for file in files:
        if file:
            p = Path(f'./uploads/{scenario}')
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
            file.save(f'./uploads/{scenario}/{file.filename}')

    id = dir2id[scenario]

    return jsonify({
        'id': id,
    }), 200


@app.route('/', methods=['GET'])
def index():
    return 'Hello, World!'

@app.route('/<path:fn>', methods=['GET'])
def server_static_files(fn):
    return send_from_directory(app.static_folder, fn)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10010)