import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Literal

import requests

from .base import Message, Storage
from .utils import extract_evoid, extract_loopid_func_name

LOG_LEVEL = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _gen_datetime(dt: datetime | None = None) -> datetime:
    """
    Generate a datetime object in UTC timezone.
    - If `dt` is None, it will return the current time in UTC.
    - If `dt` is provided, it will convert it to UTC timezone.
    """
    if dt is None:
        return datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc)


def _remove_empty_dir(path: Path):
    """
    Recursively remove empty directories.
    This function will remove the directory if it is empty after removing its subdirectories.
    """
    if path.is_dir():
        sub_dirs = [sub for sub in path.iterdir() if sub.is_dir()]
        for sub in sub_dirs:
            _remove_empty_dir(sub)

        if not any(path.iterdir()):
            path.rmdir()


class FileStorage(Storage):
    """
    The info are logginged to the file systems

    TODO: describe the storage format
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(
        self,
        obj: object,
        tag: str = "",
        timestamp: datetime | None = None,
        save_type: Literal["json", "text", "pkl"] = "pkl",
        **kwargs: Any,
    ) -> str | Path:
        # TODO: We can remove the timestamp after we implement PipeLog
        timestamp = _gen_datetime(timestamp)

        cur_p = self.path / tag.replace(".", "/")
        cur_p.mkdir(parents=True, exist_ok=True)

        path = cur_p / f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S-%f')}.log"

        if save_type == "json":
            path = path.with_suffix(".json")
            with path.open("w") as f:
                try:
                    json.dump(obj, f)
                except TypeError:
                    json.dump(json.loads(str(obj)), f)
            return path
        elif save_type == "pkl":
            path = path.with_suffix(".pkl")
            with path.open("wb") as f:
                pickle.dump(obj, f)
            return path
        elif save_type == "text":
            obj = str(obj)
            with path.open("w") as f:
                f.write(obj)
            return path

    log_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| "
        r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) *\| "
        r"(?P<caller>.+:.+:\d+) - "
    )

    def iter_msg(self, tag: str | None = None) -> Generator[Message, None, None]:
        msg_l = []

        pkl_files = "**/*.pkl" if tag is None else f"**/{tag.replace('.','/')}/**/*.pkl"
        for file in self.path.glob(pkl_files):
            if file.name == "debug_llm.pkl":
                continue
            pkl_log_tag = ".".join(file.relative_to(self.path).as_posix().replace("/", ".").split(".")[:-3])
            pid = file.parent.name

            with file.open("rb") as f:
                content = pickle.load(f)

            timestamp = datetime.strptime(file.stem, "%Y-%m-%d_%H-%M-%S-%f").replace(tzinfo=timezone.utc)

            m = Message(tag=pkl_log_tag, level="INFO", timestamp=timestamp, caller="", pid_trace=pid, content=content)

            msg_l.append(m)

        msg_l.sort(key=lambda x: x.timestamp)
        for m in msg_l:
            yield m

    def truncate(self, time: datetime) -> None:
        for file in self.path.glob("**/*.pkl"):
            timestamp = datetime.strptime(file.stem, "%Y-%m-%d_%H-%M-%S-%f").replace(tzinfo=timezone.utc)
            if timestamp > time:
                file.unlink()

        _remove_empty_dir(self.path)

    def __str__(self) -> str:
        return f"FileStorage({self.path})"


class WebStorage(Storage):
    """
    The storage for web app.
    It is used to provide the data for the web app.
    """

    def __init__(self, port: int, path: str) -> None:
        """
        Initializes the storage object with the specified port and identifier.
        Args:
            port (int): The port number to use for the storage service.
            path (str): The unique identifier for local storage, the log path.
        """
        self.url = f"http://localhost:{port}"
        self.path = path
        self.msgs = []

    def __str__(self):
        return f"WebStorage({self.url})"

    def log(self, obj: object, tag: str, timestamp: datetime | None = None, **kwargs: Any) -> str | Path:
        timestamp = _gen_datetime(timestamp)

        try:
            data = self._obj_to_json(obj=obj, tag=tag, id=self.path, timestamp=timestamp.isoformat())
            if isinstance(data, list):
                for d in data:
                    self.msgs.append(d)
            else:
                self.msgs.append(data)
            headers = {"Content-Type": "application/json"}
            requests.post(f"{self.url}/receive", json=data, headers=headers, timeout=1)
        except (requests.ConnectionError, requests.Timeout) as e:
            pass

    def truncate(self, time: datetime) -> None:
        self.msgs = [m for m in self.msgs if datetime.fromisoformat(m["msg"]["timestamp"]) <= time]

    def iter_msg(self, **kwargs: Any) -> Generator[Message, None, None]:
        for msg in self.msgs:
            yield Message(
                tag=msg["msg"]["tag"],
                level="INFO",
                timestamp=datetime.fromisoformat(msg["msg"]["timestamp"]),
                content=msg,
            )

    def _obj_to_json(
        obj: object,
        tag: str,
        id: str,
        timestamp: str,
    ) -> list[dict] | dict:
        li, fn = extract_loopid_func_name(tag)
        ei = extract_evoid(tag)
        data = {}
        if "hypothesis generation" in tag:
            from rdagent.core.proposal import Hypothesis

            h: Hypothesis = obj
            data = {
                "id": id,
                "msg": {
                    "tag": "research.hypothesis",
                    "timestamp": timestamp,
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
                    "id": id,
                    "msg": {
                        "tag": "research.tasks",
                        "timestamp": timestamp,
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
                    "id": id,
                    "msg": {
                        "tag": "research.tasks",
                        "timestamp": timestamp,
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
            from rdagent.scenarios.data_science.experiment.experiment import (
                DSExperiment,
            )

            if isinstance(obj, DSExperiment):
                from rdagent.scenarios.data_science.proposal.exp_gen.base import (
                    DSHypothesis,
                )

                h: DSHypothesis = obj.hypothesis
                tasks = [t[0] for t in obj.pending_tasks_list]
                t = tasks[0]
                data = [
                    {
                        "id": id,
                        "msg": {
                            "tag": "research.hypothesis",
                            "old_tag": tag,
                            "timestamp": timestamp,
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
                        "id": id,
                        "msg": {
                            "tag": "research.tasks",
                            "old_tag": tag,
                            "timestamp": timestamp,
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
                    "id": id,
                    "msg": {
                        "tag": "evolving.codes",
                        "timestamp": timestamp,
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
            from rdagent.components.coder.CoSTEER.evaluators import (
                CoSTEERSingleFeedback,
            )

            fl: list[CoSTEERSingleFeedback] = [i for i in obj]
            data = {
                "id": id,
                "msg": {
                    "tag": "evolving.feedbacks",
                    "timestamp": timestamp,
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
                "id": id,
                "msg": {
                    "tag": "feedback.config",
                    "timestamp": timestamp,
                    "loop_id": li,
                    "content": {"config": obj.experiment_setting},
                },
            }

        elif "Quantitative Backtesting Chart" in tag:
            import plotly

            from rdagent.log.ui.qlib_report_figure import report_figure

            data = {
                "id": id,
                "msg": {
                    "tag": "feedback.return_chart",
                    "timestamp": timestamp,
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
                        "id": id,
                        "msg": {
                            "tag": "feedback.metric",
                            "old_tag": tag,
                            "timestamp": timestamp,
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
                "id": id,
                "msg": {
                    "tag": "feedback.hypothesis_feedback",
                    "timestamp": timestamp,
                    "loop_id": li,
                    "content": content,
                },
            }

        return data
