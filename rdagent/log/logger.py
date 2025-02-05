import json
import os
import pickle
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from logging import LogRecord
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, Union

import plotly
from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

from psutil import Process

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
from rdagent.components.coder.factor_coder.evaluators import FactorSingleFeedback
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.core.utils import SingletonBaseClass

from .storage import FileStorage
from .ui.llm_st import extract_loopid_func_name
from .ui.qlib_report_figure import report_figure
from .utils import LogColors, get_caller_info


class RDAgentLog(SingletonBaseClass):
    """
    The files are organized based on the tag & PID
    Here is an example tag

    .. code-block::

        a
        - b
        - c
            - 123
              - common_logs.log
            - 1322
              - common_logs.log
            - 1233
              - <timestamp>.pkl
            - d
                - 1233-673 ...
                - 1233-4563 ...
                - 1233-365 ...

    """

    # TODO: Simplify it to introduce less concepts ( We may merge RDAgentLog, Storage &)
    # Solution:  Storage => PipeLog, View => PipeLogView, RDAgentLog is an instance of PipeLogger
    # PipeLogger.info(...) ,  PipeLogger.get_resp() to get feedback from frontend.
    # def f():
    #   logger = PipeLog()
    #   logger.info("<code>")
    #   feedback = logger.get_reps()
    _tag: str = ""

    def __init__(self, log_trace_path: Union[str, None] = RD_AGENT_SETTINGS.log_trace_path) -> None:
        if log_trace_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f")
            self.log_trace_path = Path.cwd() / "log" / timestamp
        else:
            self.log_trace_path = Path(log_trace_path)

        self.log_trace_path.mkdir(parents=True, exist_ok=True)

        self.storage = FileStorage(self.log_trace_path)

        self.main_pid = os.getpid()

        self.msgs_for_frontend = defaultdict(list)

    def set_trace_path(self, log_trace_path: str | Path) -> None:
        self.log_trace_path = Path(log_trace_path)
        self.storage = FileStorage(log_trace_path)

    @contextmanager
    def tag(self, tag: str) -> Generator[None, None, None]:
        if tag.strip() == "":
            raise ValueError("Tag cannot be empty.")
        if self._tag != "":
            tag = "." + tag

        # TODO: It may result in error in mutithreading or co-routine
        self._tag = self._tag + tag
        try:
            yield
        finally:
            self._tag = self._tag[: -len(tag)]

    def get_pids(self) -> str:
        """
        Returns a string of pids from the current process to the main process.
        Split by '-'.
        """
        pid = os.getpid()
        process = Process(pid)
        pid_chain = f"{pid}"
        while process.pid != self.main_pid:
            parent_pid = process.ppid()
            parent_process = Process(parent_pid)
            pid_chain = f"{parent_pid}-{pid_chain}"
            process = parent_process
        return pid_chain

    def file_format(self, record: "Record", raw: bool = False) -> str:
        # FIXME: the formmat is tightly coupled with the message reading in storage.
        record["message"] = LogColors.remove_ansi_codes(record["message"])
        if raw:
            return "{message}"
        return "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n"

    def format_pkl(self, base_path: str | Path):
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
                    self.msgs_for_frontend[did].append(
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
                        self.msgs_for_frontend[did].append(
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
                        self.msgs_for_frontend[did].append(
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
                    self.msgs_for_frontend[did].append(
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
                    self.msgs_for_frontend[did].append(
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
                    self.msgs_for_frontend[did].append(
                        {
                            "tag": "feedback.config",
                            "timestamp": m.timestamp.isoformat(),
                            "content": {"config": m.content.experiment_setting},
                        }
                    )

                elif "ef.Quantitative Backtesting Chart" in m.tag:
                    self.msgs_for_frontend[did].append(
                        {
                            "tag": "feedback.return_chart",
                            "timestamp": m.timestamp.isoformat(),
                            "content": {"chart_html": plotly.io.to_html(report_figure(m.content))},
                        }
                    )

                elif "model runner result" in m.tag or "factor runner result" in m.tag or "runner result" in m.tag:
                    self.msgs_for_frontend[did].append(
                        {
                            "tag": "feedback.metric",
                            "timestamp": m.timestamp.isoformat(),
                            "content": {"result": m.content.result.iloc[0]},
                        }
                    )

                elif "ef.feedback" in m.tag:
                    hf: HypothesisFeedback = m.content
                    self.msgs_for_frontend[did].append(
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
        for msgs in self.msgs_for_frontend.values():
            msgs.append({"tag": "END", "timestamp": datetime.now(timezone.utc).isoformat(), "content": {}})

        return self.msgs_for_frontend

    def log_object(self, obj: object, *, tag: str = "") -> None:
        # TODO: I think we can merge the log_object function with other normal log methods to make the interface simpler.
        caller_info = get_caller_info()
        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")

        # FIXME: it looks like a hacking... We should redesign it...
        if "debug_" in tag:
            debug_log_path = self.log_trace_path / "debug_llm.pkl"
            debug_data = {"tag": tag, "obj": obj}
            if debug_log_path.exists():
                with debug_log_path.open("rb") as f:
                    existing_data = pickle.load(f)
                existing_data.append(debug_data)
                with debug_log_path.open("wb") as f:
                    pickle.dump(existing_data, f)
            else:
                with debug_log_path.open("wb") as f:
                    pickle.dump([debug_data], f)
            return

        logp = self.storage.log(obj, name=tag, save_type="pkl")

        file_handler_id = logger.add(
            self.log_trace_path / tag.replace(".", "/") / "common_logs.log", format=self.file_format
        )
        logger.patch(lambda r: r.update(caller_info)).info(f"Logging object in {Path(logp).absolute()}")
        logger.remove(file_handler_id)

    def info(self, msg: str, *, tag: str = "", raw: bool = False) -> None:
        # TODO: too much duplicated. due to we have no logger with stream context;
        caller_info = get_caller_info()
        if raw:
            logger.remove()
            logger.add(sys.stderr, format=lambda r: "{message}")

        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")
        log_file_path = self.log_trace_path / tag.replace(".", "/") / "common_logs.log"
        if raw:
            file_handler_id = logger.add(log_file_path, format=partial(self.file_format, raw=True))
        else:
            file_handler_id = logger.add(log_file_path, format=self.file_format)

        logger.patch(lambda r: r.update(caller_info)).info(msg)
        logger.remove(file_handler_id)

        if raw:
            logger.remove()
            logger.add(sys.stderr)

    def warning(self, msg: str, *, tag: str = "") -> None:
        # TODO: reuse code
        # _log(self, msg: str, *, tag: str = "", level=Literal["warning", "error", ..]) -> None:
        # getattr(logger.patch(lambda r: r.update(caller_info)), level)(msg)
        caller_info = get_caller_info()

        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")
        file_handler_id = logger.add(
            self.log_trace_path / tag.replace(".", "/") / "common_logs.log", format=self.file_format
        )
        logger.patch(lambda r: r.update(caller_info)).warning(msg)
        logger.remove(file_handler_id)

    def error(self, msg: str, *, tag: str = "") -> None:
        caller_info = get_caller_info()

        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")
        file_handler_id = logger.add(
            self.log_trace_path / tag.replace(".", "/") / "common_logs.log", format=self.file_format
        )
        logger.patch(lambda r: r.update(caller_info)).error(msg)
        logger.remove(file_handler_id)
