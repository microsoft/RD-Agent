import inspect
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict, cast


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


def is_valid_session(log_path: Path) -> bool:
    return log_path.is_dir() and log_path.joinpath("__session__").exists()


def extract_loopid_func_name(tag: str) -> tuple[str, str] | tuple[None, None]:
    """extract loop id and function name from the tag in Message"""
    match = re.search(r"Loop_(\d+)\.([^.]+)", tag)
    return cast(tuple[str, str], match.groups()) if match else (None, None)


def extract_evoid(tag: str) -> str | None:
    """extract evo id from the tag in Message"""
    match = re.search(r"\.evo_loop_(\d+)\.", tag)
    return cast(str, match.group(1)) if match else None


def extract_json(log_content: str) -> dict | None:
    match = re.search(r"\{.*\}", log_content, re.DOTALL)
    if match:
        return cast(dict, json.loads(match.group(0)))
    return None


def log_obj_to_json(
    obj: object,
    tag: str = "",
    log_trace_path: str = None,
) -> list[dict] | dict:
    ts = datetime.now(timezone.utc).isoformat()
    li, fn = extract_loopid_func_name(tag)
    ei = extract_evoid(tag)
    data = {}
    if "hypothesis generation" in tag:
        from rdagent.core.proposal import Hypothesis

        h: Hypothesis = obj
        data = {
            "id": str(log_trace_path),
            "msg": {
                "tag": "research.hypothesis",
                "timestamp": ts,
                "loop_id": li,
                "content": {
                    "hypothesis": h.hypothesis,
                    "reason": h.reason,
                    "concise_reason": h.concise_reason,
                    "concise_justification": h.concise_justification,
                    "concise_observation": h.concise_observation,
                    "concise_knowledge": h.concise_knowledge,
                },
            },
        }

    elif "experiment generation" in tag:
        from rdagent.components.coder.factor_coder.factor import FactorTask
        from rdagent.components.coder.model_coder.model import ModelTask
        from rdagent.core.experiment import Experiment

        tasks: list[FactorTask | ModelTask] = obj
        if isinstance(tasks[0], FactorTask):
            data = {
                "id": str(log_trace_path),
                "msg": {
                    "tag": "research.tasks",
                    "timestamp": ts,
                    "loop_id": li,
                    "content": [
                        {
                            "name": t.factor_name,
                            "description": t.factor_description,
                            "formulation": t.factor_formulation,
                            "variables": t.variables,
                        }
                        for t in tasks
                    ],
                },
            }
        elif isinstance(tasks[0], ModelTask):
            data = {
                "id": str(log_trace_path),
                "msg": {
                    "tag": "research.tasks",
                    "timestamp": ts,
                    "loop_id": li,
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
                },
            }
    elif "direct_exp_gen" in tag:
        from rdagent.scenarios.data_science.experiment.experiment import DSExperiment

        if isinstance(obj, DSExperiment):
            from rdagent.scenarios.data_science.proposal.exp_gen.base import (
                DSHypothesis,
            )

            h: DSHypothesis = obj.hypothesis
            tasks = [t[0] for t in obj.pending_tasks_list]
            t = tasks[0]
            data = [
                {
                    "id": str(log_trace_path),
                    "msg": {
                        "tag": "research.hypothesis",
                        "old_tag": tag,
                        "timestamp": ts,
                        "loop_id": li,
                        "content": {
                            "name_map": {
                                "hypothesis": "RD-Agent proposes the hypothesis⬇️",
                                "concise_justification": "because the reason⬇️",
                                "concise_observation": "based on the observation⬇️",
                                "concise_knowledge": "Knowledge⬇️ gained after practice",
                                "no_hypothesis": f"No hypothesis available. Trying to construct the first runnable {h.component} component.",
                            },
                            "hypothesis": h.hypothesis,
                            "reason": h.reason,
                            "component": h.component,
                            "concise_reason": h.concise_reason,
                            "concise_justification": h.concise_justification,
                            "concise_observation": h.concise_observation,
                            "concise_knowledge": h.concise_knowledge,
                        },
                    },
                },
                {
                    "id": str(log_trace_path),
                    "msg": {
                        "tag": "research.tasks",
                        "old_tag": tag,
                        "timestamp": ts,
                        "loop_id": li,
                        "content": [
                            (
                                {
                                    "name": t.name,
                                    "description": t.description,
                                }
                                if not hasattr(t, "architecture")
                                else {
                                    "name": t.name,
                                    "description": t.description,
                                    "model_type": t.model_type,
                                    "architecture": t.architecture,
                                    "hyperparameters": t.hyperparameters,
                                }
                            )
                        ],
                    },
                },
            ]
    elif f"evo_loop_{ei}.evolving code" in tag and "coding" in tag:
        from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace
        from rdagent.components.coder.model_coder.model import (
            ModelFBWorkspace,
            ModelTask,
        )
        from rdagent.core.experiment import FBWorkspace

        ws: list[FactorFBWorkspace | ModelFBWorkspace] = [i for i in obj]
        if all(isinstance(item, FactorFBWorkspace) for item in ws) or all(
            isinstance(item, ModelFBWorkspace) for item in ws
        ):
            data = {
                "id": str(log_trace_path),
                "msg": {
                    "tag": "evolving.codes",
                    "timestamp": ts,
                    "loop_id": li,
                    "evo_id": ei,
                    "content": [
                        {
                            "target_task_name": w.target_task.name,
                            "codes": w.file_dict,
                        }
                        for w in ws
                        if w
                    ],
                },
            }
    elif f"evo_loop_{ei}.evolving feedback" in tag and "coding" in tag:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback

        fl: list[CoSTEERSingleFeedback] = [i for i in obj]
        data = {
            "id": str(log_trace_path),
            "msg": {
                "tag": "evolving.feedbacks",
                "timestamp": ts,
                "loop_id": li,
                "evo_id": ei,
                "content": [
                    {
                        "final_decision": f.final_decision,
                        # "final_feedback": f.final_feedback,
                        "execution": f.execution,
                        "code": f.code,
                        "return_checking": f.return_checking,
                    }
                    for f in fl
                    if f
                ],
            },
        }
    elif "scenario" in tag:
        data = {
            "id": str(log_trace_path),
            "msg": {
                "tag": "feedback.config",
                "timestamp": ts,
                "loop_id": li,
                "content": {"config": obj.experiment_setting},
            },
        }

    elif "Quantitative Backtesting Chart" in tag:
        import plotly

        from rdagent.log.ui.qlib_report_figure import report_figure

        data = {
            "id": str(log_trace_path),
            "msg": {
                "tag": "feedback.return_chart",
                "timestamp": ts,
                "loop_id": li,
                "content": {"chart_html": plotly.io.to_html(report_figure(obj))},
            },
        }
    elif "running" in tag:
        from rdagent.core.experiment import Experiment

        if isinstance(obj, Experiment):
            if obj.result is not None:
                result_str = obj.result.to_json()
                data = {
                    "id": str(log_trace_path),
                    "msg": {
                        "tag": "feedback.metric",
                        "old_tag": tag,
                        "timestamp": ts,
                        "loop_id": li,
                        "content": {
                            "result": result_str,
                        },
                    },
                }
    elif "feedback" in tag:
        from rdagent.core.proposal import ExperimentFeedback, HypothesisFeedback

        ef: ExperimentFeedback = obj
        content = (
            {
                "observations": ef.observations,
                "hypothesis_evaluation": ef.hypothesis_evaluation,
                "new_hypothesis": ef.new_hypothesis,
                "decision": ef.decision,
                "reason": ef.reason,
                "exception": ef.exception,
            }
            if isinstance(ef, HypothesisFeedback)
            else {
                "decision": ef.decision,
                "reason": ef.reason,
                "exception": ef.exception,
            }
        )
        data = {
            "id": str(log_trace_path),
            "msg": {
                "tag": "feedback.hypothesis_feedback",
                "timestamp": ts,
                "loop_id": li,
                "content": content,
            },
        }

    return data
