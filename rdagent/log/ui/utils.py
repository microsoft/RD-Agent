import typing
import requests
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import plotly

from rdagent.log.ui.llm_st import extract_evoid, extract_loopid_func_name
from rdagent.log.ui.qlib_report_figure import report_figure

if typing.TYPE_CHECKING:
    from rdagent.components.coder.CoSTEER.evaluators import (
        CoSTEERSingleFeedbackDeprecated,
    )
    from rdagent.components.coder.factor_coder.evaluators import FactorSingleFeedback
    from rdagent.components.coder.factor_coder.factor import (
        FactorFBWorkspace,
        FactorTask,
    )
    from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
    from rdagent.core.experiment import Experiment
    from rdagent.core.proposal import Hypothesis, HypothesisFeedback


msgs_for_frontend = defaultdict(list)


def format_pkl(
    obj: object,
    tag: str = "",
    log_trace_path: str = None,
    url: str = "http://localhost:5000/receive",
    headers: dict = {'Content-Type': 'application/json'},
):

    ts = datetime.now(timezone.utc).isoformat()
    lp = extract_loopid_func_name(tag)
    lp_id = lp[0] if lp and lp[0] is not None else None

    if "r.hypothesis generation" in tag:
        h: Hypothesis = obj
        data = {
            "id": log_trace_path,
            "msg": {
                "tag": "research.hypothesis",
                "timestamp": ts,
                "content": {
                    "name_map": {
                        "hypothesis": "RD-Agent proposes the hypothesis⬇️",
                        "concise_justification": "because the reason⬇️",
                        "concise_observation": "based on the observation⬇️",
                        "concise_knowledge": "Knowledge⬇️ gained after practice",
                    },
                    "hypothesis": h.hypothesis,
                    "concise_justification": h.concise_justification,
                    "concise_observation": h.concise_observation,
                    "concise_knowledge": h.concise_knowledge,
                },
            }
        }
        response = requests.post(url, json=data, headers=headers)

    elif "r.experiment generation" in tag or "d.load_experiment" in tag:
        if "d.load_experiment" in tag:
            if isinstance(obj, Experiment):
                tasks: list[FactorTask | ModelTask] = obj.sub_tasks
        else:
            tasks: list[FactorTask | ModelTask] = obj
        if isinstance(tasks[0], FactorTask):
            data = {
                "id": log_trace_path,
                "msg": {
                    "tag": "research.tasks",
                    "timestamp": ts,
                    "content": [
                        {
                            "name": t.factor_name,
                            "description": t.factor_description,
                            "formulation": t.factor_formulation,
                            "variables": t.variables,
                        }
                        for t in tasks
                    ],
                }
            }
        elif isinstance(tasks[0], ModelTask):
            data = {
                "id": log_trace_path,
                "msg": {
                    "tag": "research.tasks",
                    "timestamp": ts,
                    "content": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "model_type": t.model_type,
                            "formulation": t.formulation,
                            "variables": t.variables,
                        }
                        for t in tasks
                    ],
                }
            }
        response = requests.post(url, json=data, headers=headers)

    elif f"evo_loop_{lp_id}.evolving code" in tag:
        ws: list[FactorFBWorkspace | ModelFBWorkspace] = [i for i in obj]
        data = {
            "id": log_trace_path,
            "msg": {
                "tag": "evolving.codes",
                "timestamp": ts,
                "content": [
                    {
                        "target_task_name": (
                            w.target_task.name if isinstance(w.target_task, ModelTask) else w.target_task.factor_name
                        ),
                        "code": w.file_dict,
                    }
                    for w in ws
                    if w
                ],
            }
        }
        response = requests.post(url, json=data, headers=headers)

    elif f"evo_loop_{lp_id}.evolving feedback" in tag:
        fl: list[FactorSingleFeedback | CoSTEERSingleFeedbackDeprecated] = [i for i in obj]
        data = {
            "id": log_trace_path,
            "msg": {
                "tag": "evolving.feedbacks",
                "timestamp": ts,
                "content": [
                    {
                        "final_decision": f.final_decision,
                        "final_feedback": f.final_feedback,
                        "execution_feedback": f.execution_feedback,
                        "code_feedback": f.code_feedback,
                        "value_feedback": (
                            f.value_feedback
                            if isinstance(f, CoSTEERSingleFeedbackDeprecated)
                            else f.factor_value_feedback
                        ),
                        "model_shape_feedback": (
                            f.shape_feedback if isinstance(f, CoSTEERSingleFeedbackDeprecated) else None
                        ),
                    }
                    for f in fl
                    if f
                ],
            }
        }
        response = requests.post(url, json=data, headers=headers)

    elif "scenario" in tag:
        data = {
            "id": log_trace_path,
            "msg": {"tag": "feedback.config", "timestamp": ts, "content": {"config": obj.experiment_setting}}
        }
        response = requests.post(url, json=data, headers=headers)

    elif "ef.Quantitative Backtesting Chart" in tag:
        data = {
            "id": log_trace_path,
            "msg": {
                "tag": "feedback.return_chart",
                "timestamp": ts,
                "content": {"chart_html": plotly.io.to_html(report_figure(obj))},
            }
        }
        response = requests.post(url, json=data, headers=headers)

    elif "model runner result" in tag or "factor runner result" in tag or "runner result" in tag:
        if isinstance(obj, Experiment):
            data = {
                "id": log_trace_path,
                "msg":     {"tag": "feedback.metric", "timestamp": ts, "content": {"result": obj.result.iloc[0]}}
            }
            response = requests.post(url, json=data, headers=headers)

    elif "ef.feedback" in tag:
        hf: HypothesisFeedback = obj
        data = {
            "id": log_trace_path,
            "msg": {
                "tag": "feedback.hypothesis_feedback",
                "timestamp": ts,
                "content": {
                    "observations": hf.observations,
                    "hypothesis_evaluation": hf.hypothesis_evaluation,
                    "new_hypothesis": hf.new_hypothesis,
                    "decision": hf.decision,
                    "reason": hf.reason,
                },
            }
        }
        response = requests.post(url, json=data, headers=headers)
    # for msgs in msgs_for_frontend.values():
    #     msgs.append({"tag": "END", "timestamp": ts, "content": {}})
