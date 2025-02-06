import inspect
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, TypedDict, Union

import plotly

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
from rdagent.components.coder.factor_coder.evaluators import FactorSingleFeedback
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.log.storage import FileStorage
from rdagent.log.ui.llm_st import extract_evoid, extract_loopid_func_name
from rdagent.log.ui.qlib_report_figure import report_figure

msgs_for_frontend = defaultdict(list)


class LogColors:
    """
    ANSI color codes for use in console output.
    """

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BLACK = "\033[30m"

    BOLD = "\033[1m"
    ITALIC = "\033[3m"

    END = "\033[0m"

    @classmethod
    def get_all_colors(cls: type["LogColors"]) -> list:
        names = dir(cls)
        names = [name for name in names if not name.startswith("__") and not callable(getattr(cls, name))]
        return [getattr(cls, name) for name in names]

    def render(self, text: str, color: str = "", style: str = "") -> str:
        """
        render text by input color and style.
        It's not recommend that input text is already rendered.
        """
        # This method is called too frequently, which is not good.
        colors = self.get_all_colors()
        # Perhaps color and font should be distinguished here.
        if color and color in colors:
            error_message = f"color should be in: {colors} but now is: {color}"
            raise ValueError(error_message)
        if style and style in colors:
            error_message = f"style should be in: {colors} but now is: {style}"
            raise ValueError(error_message)

        text = f"{color}{text}{self.END}"

        return f"{style}{text}{self.END}"

    @staticmethod
    def remove_ansi_codes(s: str) -> str:
        """
        It is for removing ansi ctrl characters in the string(e.g. colored text)
        """
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        return ansi_escape.sub("", s)


class CallerInfo(TypedDict):
    function: str
    line: int
    name: Optional[str]


def get_caller_info() -> CallerInfo:
    # Get the current stack information
    stack = inspect.stack()
    # The second element is usually the caller's information
    caller_info = stack[2]
    frame = caller_info[0]
    info: CallerInfo = {
        "line": caller_info.lineno,
        "name": frame.f_globals["__name__"],  # Get the module name from the frame's globals
        "function": frame.f_code.co_name,  # Get the caller's function name
    }
    return info


def format_pkl(base_path: str | Path):
    base_path = Path(base_path)
    dir2id = {dir_name.name: idx for idx, dir_name in enumerate(base_path.iterdir())}
    for dn, did in dir2id.items():
        fs = FileStorage(base_path / dn)
        for m in fs.iter_msg():
            lp = extract_loopid_func_name(m.tag)
            lp_id = lp[0] if lp and lp[0] is not None else None
            # lp_id = (lp := extract_loopid_func_name(m.tag))[0] if lp[0] is not None else None
            if "r.hypothesis generation" in m.tag:
                h: Hypothesis = m.content
                msgs_for_frontend[did].append(
                    {
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
                            "concise_justification": h.concise_justification,
                            "concise_observation": h.concise_observation,
                            "concise_knowledge": h.concise_knowledge,
                        },
                    }
                )

            # m.tag 中不存在 d.load_experiment, 存在 r.load_experiment
            elif "r.experiment generation" in m.tag or "d.load_experiment" in m.tag:
                if "d.load_experiment" in m.tag:
                    tasks: list[FactorTask | ModelTask] = m.content.sub_tasks
                else:
                    tasks: list[FactorTask | ModelTask] = m.content
                if isinstance(tasks[0], FactorTask):
                    msgs_for_frontend[did].append(
                        {
                            "tag": "research.tasks",
                            "timestamp": m.timestamp.isoformat(),
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
                    )
                elif isinstance(tasks[0], ModelTask):
                    msgs_for_frontend[did].append(
                        {
                            "tag": "research.tasks",
                            "timestamp": m.timestamp.isoformat(),
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
                    )

            elif f"evo_loop_{lp_id}.evolving code" in m.tag:
                ws: list[FactorFBWorkspace | ModelFBWorkspace] = [i for i in m.content]
                msgs_for_frontend[did].append(
                    {
                        "tag": "evolving.codes",
                        "timestamp": m.timestamp.isoformat(),
                        "content": [
                            {
                                "target_task_name": (
                                    w.target_task.name
                                    if isinstance(w.target_task, ModelTask)
                                    else w.target_task.factor_name
                                ),
                                "code": w.file_dict,
                            }
                            for w in ws
                            if w
                        ],
                    }
                )

            elif f"evo_loop_{lp_id}.evolving feedback" in m.tag:
                fl: list[FactorSingleFeedback | CoSTEERSingleFeedbackDeprecated] = [i for i in m.content]
                msgs_for_frontend[did].append(
                    {
                        "tag": "evolving.feedbacks",
                        "timestamp": m.timestamp.isoformat(),
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
                )

            elif "scenario" in m.tag:
                msgs_for_frontend[did].append(
                    {
                        "tag": "feedback.config",
                        "timestamp": m.timestamp.isoformat(),
                        "content": {"config": m.content.experiment_setting},
                    }
                )

            elif "ef.Quantitative Backtesting Chart" in m.tag:
                msgs_for_frontend[did].append(
                    {
                        "tag": "feedback.return_chart",
                        "timestamp": m.timestamp.isoformat(),
                        "content": {"chart_html": plotly.io.to_html(report_figure(m.content))},
                    }
                )

            elif "model runner result" in m.tag or "factor runner result" in m.tag or "runner result" in m.tag:
                msgs_for_frontend[did].append(
                    {
                        "tag": "feedback.metric",
                        "timestamp": m.timestamp.isoformat(),
                        "content": {"result": m.content.result.iloc[0]},
                    }
                )

            elif "ef.feedback" in m.tag:
                hf: HypothesisFeedback = m.content
                msgs_for_frontend[did].append(
                    {
                        "tag": "feedback.hypothesis_feedback",
                        "timestamp": m.timestamp.isoformat(),
                        "content": {
                            "observations": hf.observations,
                            "hypothesis_evaluation": hf.hypothesis_evaluation,
                            "new_hypothesis": hf.new_hypothesis,
                            "decision": hf.decision,
                            "reason": hf.reason,
                        },
                    }
                )
    for msgs in msgs_for_frontend.values():
        msgs.append({"tag": "END", "timestamp": datetime.now(timezone.utc).isoformat(), "content": {}})

    return msgs_for_frontend
