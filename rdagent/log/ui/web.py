import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Callable, Type

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from rdagent.components.coder.factor_coder.evaluators import FactorSingleFeedback
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.components.coder.model_coder.evaluators import ModelSingleFeedback
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.proposal import Hypothesis, HypothesisFeedback, Trace
from rdagent.log.base import Message, Storage, View
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import (
    QlibModelExperiment,
    QlibModelScenario,
)

st.set_page_config(layout="wide")

TIME_DELAY = 0.001


class WebView(View):
    def __init__(self, ui: "StWindow"):
        self.ui = ui
        # Save logs to your desired data structure
        # ...

    def display(self, s: Storage, watch: bool = False):
        for msg in s.iter_msg():  # iterate overtime
            # NOTE:  iter_msg will correctly separate the information.
            # TODO: msg may support streaming mode.
            self.ui.consume_msg(msg)


class StWindow:
    def __init__(self, container: "DeltaGenerator"):
        self.container = container

    def consume_msg(self, msg: Message):
        msg_str = f"{msg.timestamp.astimezone(timezone.utc).isoformat()} | {msg.level} | {msg.caller} - {msg.content}"
        self.container.code(msg_str, language="log")


class LLMWindow(StWindow):
    def __init__(self, container: "DeltaGenerator", session_name: str = "common"):
        self.session_name = session_name
        self.container = container.expander(f"{self.session_name} message")

    def consume_msg(self, msg: Message):
        self.container.chat_message("user").markdown(f"{msg.content}")


class ProgressTabsWindow(StWindow):
    """
    For windows with stream messages, will refresh when a new tab is created.
    """

    def __init__(
        self,
        container: "DeltaGenerator",
        inner_class: Type[StWindow] = StWindow,
        mapper: Callable[[Message], str] = lambda x: x.pid_trace,
    ):
        self.inner_class = inner_class
        self.mapper = mapper

        self.container = container.empty()
        self.tab_windows: dict[str, StWindow] = defaultdict(None)
        self.tab_caches: dict[str, list[Message]] = defaultdict(list)

    def consume_msg(self, msg: Message):
        name = self.mapper(msg)

        if name not in self.tab_windows:
            # new tab need to be created, current streamlit container need to be updated.
            names = list(self.tab_windows.keys()) + [name]

            if len(names) == 1:
                tabs = [self.container.container()]
            else:
                tabs = self.container.tabs(names)

            for id, name in enumerate(names):
                self.tab_windows[name] = self.inner_class(tabs[id])

            # consume the cache
            for name in self.tab_caches:
                for msg in self.tab_caches[name]:
                    self.tab_windows[name].consume_msg(msg)

        self.tab_caches[name].append(msg)
        self.tab_windows[name].consume_msg(msg)


class ObjectsTabsWindow(StWindow):
    def __init__(
        self,
        container: "DeltaGenerator",
        inner_class: Type[StWindow] = StWindow,
        mapper: Callable[[object], str] = lambda x: str(x),
        tab_names: list[str] | None = None,
    ):
        self.inner_class = inner_class
        self.mapper = mapper
        self.container = container
        self.tab_names = tab_names

    def consume_msg(self, msg: Message):
        if isinstance(msg.content, list):
            if self.tab_names:
                assert len(self.tab_names) == len(
                    msg.content
                ), "List of objects should have the same length as provided tab names."
                objs_dict = {self.tab_names[id]: obj for id, obj in enumerate(msg.content)}
            else:
                objs_dict = {self.mapper(obj): obj for obj in msg.content}
        elif not isinstance(msg.content, dict):
            raise ValueError("Message content should be a list or a dict of objects.")

        # two many tabs may cause display problem
        tab_names = list(objs_dict.keys())
        tabs = []
        for i in range(0, len(tab_names), 10):
            tabs.extend(self.container.tabs(tab_names[i : i + 10]))

        for id, obj in enumerate(objs_dict.values()):
            splited_msg = Message(
                tag=msg.tag,
                level=msg.level,
                timestamp=msg.timestamp,
                caller=msg.caller,
                pid_trace=msg.pid_trace,
                content=obj,
            )
            self.inner_class(tabs[id]).consume_msg(splited_msg)


class RoundTabsWindow(StWindow):
    def __init__(
        self,
        container: "DeltaGenerator",
        new_tab_func: Callable[[Message], bool],
        inner_class: Type[StWindow] = StWindow,
        title: str = "Round tabs",
    ):
        container.markdown(f"### **{title}**")
        self.inner_class = inner_class
        self.new_tab_func = new_tab_func
        self.round = 0

        self.current_win = StWindow(container)
        self.tabs_c = container.empty()

    def consume_msg(self, msg: Message):
        if self.new_tab_func(msg):
            self.round += 1
            self.current_win = self.inner_class(self.tabs_c.tabs([str(i) for i in range(1, self.round + 1)])[-1])

        self.current_win.consume_msg(msg)


class HypothesisWindow(StWindow):
    def consume_msg(self, msg: Message | Hypothesis):
        h: Hypothesis = msg.content if isinstance(msg, Message) else msg

        self.container.markdown("#### **Hypothesis💡**")
        self.container.markdown(
            f"""
- **Hypothesis**: {h.hypothesis}
- **Reason**: {h.reason}"""
        )


class HypothesisFeedbackWindow(StWindow):
    def consume_msg(self, msg: Message | HypothesisFeedback):
        h: HypothesisFeedback = msg.content if isinstance(msg, Message) else msg

        self.container.markdown("#### **Hypothesis Feedback🔍**")
        self.container.markdown(
            f"""
- **Observations**: {h.observations}
- **Hypothesis Evaluation**: {h.hypothesis_evaluation}
- **New Hypothesis**: {h.new_hypothesis}
- **Decision**: {h.decision}
- **Reason**: {h.reason}"""
        )


class FactorTaskWindow(StWindow):
    def consume_msg(self, msg: Message | FactorTask):
        ft: FactorTask = msg.content if isinstance(msg, Message) else msg

        self.container.markdown(f"**Factor Name**: {ft.factor_name}")
        self.container.markdown(f"**Description**: {ft.factor_description}")
        self.container.latex(f"Formulation: {ft.factor_formulation}")

        variables_df = pd.DataFrame(ft.variables, index=["Description"]).T
        variables_df.index.name = "Variable"
        self.container.table(variables_df)
        self.container.text(f"Factor resources: {ft.factor_resources}")


class ModelTaskWindow(StWindow):
    def consume_msg(self, msg: Message | ModelTask):
        mt: ModelTask = msg.content if isinstance(msg, Message) else msg

        self.container.markdown(f"**Model Name**: {mt.name}")
        self.container.markdown(f"**Model Type**: {mt.model_type}")
        self.container.markdown(f"**Description**: {mt.description}")
        self.container.latex(f"Formulation: {mt.formulation}")

        variables_df = pd.DataFrame(mt.variables, index=["Value"]).T
        variables_df.index.name = "Variable"
        self.container.table(variables_df)


class FactorFeedbackWindow(StWindow):
    def consume_msg(self, msg: Message | FactorSingleFeedback):
        fb: FactorSingleFeedback = msg.content if isinstance(msg, Message) else msg

        self.container.markdown(
            f"""### :blue[Factor Execution Feedback]
{fb.execution_feedback}
### :blue[Factor Code Feedback]
{fb.code_feedback}
### :blue[Factor Value Feedback]
{fb.value_feedback}
### :blue[Factor Final Feedback]
{fb.final_feedback}
### :blue[Factor Final Decision]
This implementation is {'SUCCESS' if fb.final_decision else 'FAIL'}.
"""
        )


class ModelFeedbackWindow(StWindow):
    def consume_msg(self, msg: Message | ModelSingleFeedback):
        mb: ModelSingleFeedback = msg.content if isinstance(msg, Message) else msg

        self.container.markdown(
            f"""### :blue[Model Execution Feedback]
{mb.execution_feedback}
### :blue[Model Shape Feedback]
{mb.shape_feedback}
### :blue[Model Value Feedback]
{mb.value_feedback}
### :blue[Model Code Feedback]
{mb.code_feedback}
### :blue[Model Final Feedback]
{mb.final_feedback}
### :blue[Model Final Decision]
This implementation is {'SUCCESS' if mb.final_decision else 'FAIL'}.
"""
        )


class WorkspaceWindow(StWindow):
    def __init__(self, container: "DeltaGenerator", show_task_info: bool = False):
        self.container = container
        self.show_task_info = show_task_info

    def consume_msg(self, msg: Message | FactorFBWorkspace | ModelFBWorkspace):
        ws: FactorFBWorkspace | ModelFBWorkspace = msg.content if isinstance(msg, Message) else msg

        # no workspace
        if ws is None:
            return

        # task info
        if self.show_task_info:
            task_msg = deepcopy(msg)
            task_msg.content = ws.target_task
            if isinstance(ws, FactorFBWorkspace):
                self.container.subheader("Factor Info")
                FactorTaskWindow(self.container.container()).consume_msg(task_msg)
            else:
                self.container.subheader("Model Info")
                ModelTaskWindow(self.container.container()).consume_msg(task_msg)

        # task codes
        for k, v in ws.code_dict.items():
            self.container.markdown(f"`{k}`")
            self.container.code(v, language="python")


class QlibFactorExpWindow(StWindow):
    def __init__(self, container: DeltaGenerator, show_task_info: bool = False):
        self.container = container
        self.show_task_info = show_task_info

    def consume_msg(self, msg: Message | QlibFactorExperiment):
        exp: QlibFactorExperiment = msg.content if isinstance(msg, Message) else msg

        # factor tasks
        if self.show_task_info:
            ftm_msg = deepcopy(msg)
            ftm_msg.content = [ws for ws in exp.sub_workspace_list if ws]
            self.container.markdown("**Factor Tasks**")
            ObjectsTabsWindow(
                self.container.container(),
                inner_class=WorkspaceWindow,
                mapper=lambda x: x.target_task.factor_name,
            ).consume_msg(ftm_msg)

        # result
        self.container.markdown("**Results**")
        results = pd.DataFrame({f"base_exp_{id}": e.result for id, e in enumerate(exp.based_experiments)})
        results["now"] = exp.result

        self.container.expander("results table").table(results)

        try:
            bar_chart = px.bar(results, orientation="h", barmode="group")
            self.container.expander("results chart").plotly_chart(bar_chart)
        except:
            self.container.text("Results are incomplete.")


class QlibModelExpWindow(StWindow):
    def __init__(self, container: DeltaGenerator, show_task_info: bool = False):
        self.container = container
        self.show_task_info = show_task_info

    def consume_msg(self, msg: Message | QlibModelExperiment):
        exp: QlibModelExperiment = msg.content if isinstance(msg, Message) else msg

        # model tasks
        if self.show_task_info:
            _msg = deepcopy(msg)
            _msg.content = [ws for ws in exp.sub_workspace_list if ws]
            self.container.markdown("**Model Tasks**")
            ObjectsTabsWindow(
                self.container.container(),
                inner_class=WorkspaceWindow,
                mapper=lambda x: x.target_task.name,
            ).consume_msg(_msg)

        # result
        self.container.subheader("Results", divider=True)
        results = pd.DataFrame({f"base_exp_{id}": e.result for id, e in enumerate(exp.based_experiments)})
        results["now"] = exp.result

        self.container.expander("results table").table(results)


class SimpleTraceWindow(StWindow):
    def __init__(
        self, container: "DeltaGenerator" = st.container(), show_llm: bool = False, show_common_logs: bool = False
    ):
        super().__init__(container)
        self.show_llm = show_llm
        self.show_common_logs = show_common_logs
        self.pid_trace = ""
        self.current_tag = ""

        self.current_win = StWindow(self.container)
        self.evolving_tasks: list[str] = []

    def consume_msg(self, msg: Message):
        # divide tag levels
        if len(msg.tag) > len(self.current_tag):
            # write a header about current task, if it is llm message, not write.
            if not msg.tag.endswith("llm_messages"):
                self.container.header(msg.tag.replace(".", " ➡ "), divider=True)

        self.current_tag = msg.tag

        # set log writer (window) according to msg
        if msg.tag.endswith("llm_messages"):
            # llm messages logs
            if not self.show_llm:
                return
            if not isinstance(self.current_win, LLMWindow):
                self.current_win = LLMWindow(self.container)
        elif isinstance(msg.content, Hypothesis):
            # hypothesis
            self.current_win = HypothesisWindow(self.container)
        elif isinstance(msg.content, HypothesisFeedback):
            # hypothesis feedback
            self.current_win = HypothesisFeedbackWindow(self.container)
        elif isinstance(msg.content, QlibFactorExperiment):
            self.current_win = QlibFactorExpWindow(self.container)
        elif isinstance(msg.content, QlibModelExperiment):
            self.current_win = QlibModelExpWindow(self.container)
        elif isinstance(msg.content, list):
            msg.content = [m for m in msg.content if m]
            if len(msg.content) == 0:
                return
            if isinstance(msg.content[0], FactorTask):
                self.current_win = ObjectsTabsWindow(
                    self.container.expander("Factor Tasks"), FactorTaskWindow, lambda x: x.factor_name
                )
            elif isinstance(msg.content[0], ModelTask):
                self.current_win = ObjectsTabsWindow(
                    self.container.expander("Model Tasks"), ModelTaskWindow, lambda x: x.name
                )

            elif isinstance(msg.content[0], FactorFBWorkspace):
                self.current_win = ObjectsTabsWindow(
                    self.container.expander("Factor Workspaces"),
                    inner_class=WorkspaceWindow,
                    mapper=lambda x: x.target_task.factor_name,
                )
                self.evolving_tasks = [m.target_task.factor_name for m in msg.content]
            elif isinstance(msg.content[0], ModelFBWorkspace):
                self.current_win = ObjectsTabsWindow(
                    self.container.expander("Model Workspaces"),
                    inner_class=WorkspaceWindow,
                    mapper=lambda x: x.target_task.name,
                )
                self.evolving_tasks = [m.target_task.name for m in msg.content]

            elif isinstance(msg.content[0], FactorSingleFeedback):
                self.current_win = ObjectsTabsWindow(
                    self.container.expander("Factor Feedbacks"),
                    inner_class=FactorFeedbackWindow,
                    tab_names=self.evolving_tasks,
                )
            elif isinstance(msg.content[0], ModelSingleFeedback):
                self.current_win = ObjectsTabsWindow(
                    self.container.expander("Model Feedbacks"),
                    inner_class=ModelFeedbackWindow,
                    tab_names=self.evolving_tasks,
                )
        else:
            # common logs
            if not self.show_common_logs:
                return
            self.current_win = StWindow(self.container)

        self.current_win.consume_msg(msg)


def mock_msg(obj) -> Message:
    return Message(tag="mock", level="INFO", timestamp=datetime.now(), pid_trace="000", caller="mock", content=obj)


class TraceObjWindow(StWindow):
    def __init__(self, container: "DeltaGenerator" = st.container()):
        self.container = container

    def consume_msg(self, msg: Message | Trace):
        if isinstance(msg, Message):
            trace: Trace = msg.content
        else:
            trace = msg

        for id, (h, e, hf) in enumerate(trace.hist):
            self.container.header(f"Trace History {id}", divider=True)
            HypothesisWindow(self.container).consume_msg(mock_msg(h))
            if isinstance(e, QlibFactorExperiment):
                QlibFactorExpWindow(self.container).consume_msg(mock_msg(e))
            else:
                QlibModelExpWindow(self.container).consume_msg(mock_msg(e))
            HypothesisFeedbackWindow(self.container).consume_msg(mock_msg(hf))


class ResearchWindow(StWindow):
    def consume_msg(self, msg: Message):
        if msg.tag.endswith("hypothesis generation"):
            HypothesisWindow(self.container.container()).consume_msg(msg)
        elif msg.tag.endswith("experiment generation"):
            if isinstance(msg.content, list):
                if isinstance(msg.content[0], FactorTask):
                    self.container.markdown("**Factor Tasks**")
                    ObjectsTabsWindow(
                        self.container.container(), FactorTaskWindow, lambda x: x.factor_name
                    ).consume_msg(msg)
                elif isinstance(msg.content[0], ModelTask):
                    self.container.markdown("**Model Tasks**")
                    ObjectsTabsWindow(self.container.container(), ModelTaskWindow, lambda x: x.name).consume_msg(msg)
        elif msg.tag.endswith("load_pdf_screenshot"):
            self.container.image(msg.content)
        elif msg.tag.endswith("load_factor_tasks"):
            self.container.json(msg.content)


class EvolvingWindow(StWindow):
    def __init__(self, container: "DeltaGenerator"):
        self.container = container
        self.evolving_tasks: list[str] = []

    def consume_msg(self, msg: Message):
        if msg.tag.endswith("evolving code"):
            if isinstance(msg.content, list):
                msg.content = [m for m in msg.content if m]
                if len(msg.content) == 0:
                    return
                if isinstance(msg.content[0], FactorFBWorkspace):
                    self.container.markdown("**Factor Codes**")
                    ObjectsTabsWindow(
                        self.container.container(),
                        inner_class=WorkspaceWindow,
                        mapper=lambda x: x.target_task.factor_name,
                    ).consume_msg(msg)
                    self.evolving_tasks = [m.target_task.factor_name for m in msg.content]
                elif isinstance(msg.content[0], ModelFBWorkspace):
                    self.container.markdown("**Model Codes**")
                    ObjectsTabsWindow(
                        self.container.container(), inner_class=WorkspaceWindow, mapper=lambda x: x.target_task.name
                    ).consume_msg(msg)
                    self.evolving_tasks = [m.target_task.name for m in msg.content]
        elif msg.tag.endswith("evolving feedback"):
            if isinstance(msg.content, list):
                msg.content = [m for m in msg.content if m]
                if len(msg.content) == 0:
                    return
                if isinstance(msg.content[0], FactorSingleFeedback):
                    self.container.markdown("**Factor Feedbacks🔍**")
                    ObjectsTabsWindow(
                        self.container.container(), inner_class=FactorFeedbackWindow, tab_names=self.evolving_tasks
                    ).consume_msg(msg)
                elif isinstance(msg.content[0], ModelSingleFeedback):
                    self.container.markdown("**Model Feedbacks🔍**")
                    ObjectsTabsWindow(
                        self.container.container(), inner_class=ModelFeedbackWindow, tab_names=self.evolving_tasks
                    ).consume_msg(msg)


class DevelopmentWindow(StWindow):
    def __init__(self, container: "DeltaGenerator"):
        self.E_win = RoundTabsWindow(
            container.container(),
            new_tab_func=lambda x: x.tag.endswith("evolving code"),
            inner_class=EvolvingWindow,
            title="Evolving Loops🔧",
        )

    def consume_msg(self, msg: Message):
        if "evolving" in msg.tag:
            self.E_win.consume_msg(msg)


class FeedbackWindow(StWindow):
    def __init__(self, container: "DeltaGenerator"):
        self.container = container

    def consume_msg(self, msg: Message):
        if msg.tag.endswith("returns"):
            fig = px.line(msg.content)
            self.container.markdown("**Returns📈**")
            self.container.plotly_chart(fig)
        elif isinstance(msg.content, HypothesisFeedback):
            HypothesisFeedbackWindow(self.container.container(border=True)).consume_msg(msg)
        elif isinstance(msg.content, QlibModelExperiment):
            QlibModelExpWindow(self.container.container(border=True)).consume_msg(msg)
        elif isinstance(msg.content, QlibFactorExperiment):
            QlibFactorExpWindow(self.container.container(border=True)).consume_msg(msg)


class SingleRDLoopWindow(StWindow):
    def __init__(self, container: "DeltaGenerator"):
        self.container = container
        col1, col2 = self.container.columns([2, 3])
        self.R_win = ResearchWindow(col1.container(border=True))
        self.F_win = FeedbackWindow(col1.container(border=True))
        self.D_win = DevelopmentWindow(col2.container(border=True))

    def consume_msg(self, msg: Message):
        tags = msg.tag.split(".")
        if "r" in tags:
            self.R_win.consume_msg(msg)
        elif "d" in tags:
            self.D_win.consume_msg(msg)
        elif "ef" in tags:
            self.F_win.consume_msg(msg)


class TraceWindow(StWindow):
    def __init__(
        self, container: "DeltaGenerator" = st.container(), show_llm: bool = False, show_common_logs: bool = False
    ):
        self.show_llm = show_llm
        self.show_common_logs = show_common_logs
        image_c, scen_c = container.columns([2, 3], vertical_alignment="center")
        image_c.image("scen.png")
        scen_c.container(border=True).markdown(QlibModelScenario().rich_style_description)
        top_container = container.container()
        col1, col2 = top_container.columns([2, 3])
        chart_c = col2.container(border=True, height=500)
        chart_c.markdown("**Metrics📈**")
        self.chart_c = chart_c.empty()
        hypothesis_status_c = col1.container(border=True, height=500)
        hypothesis_status_c.markdown("**Hypotheses🏅**")
        self.summary_c = hypothesis_status_c.empty()

        self.RDL_win = RoundTabsWindow(
            container.container(),
            new_tab_func=lambda x: x.tag.endswith("hypothesis generation"),
            inner_class=SingleRDLoopWindow,
            title="R&D Loops♾️",
        )

        self.hypothesis_decisions = defaultdict(bool)
        self.hypotheses: list[Hypothesis] = []

        self.results = []

    def consume_msg(self, msg: Message):
        if not self.show_llm and "llm_messages" in msg.tag:
            return
        if not self.show_common_logs and isinstance(msg.content, str):
            return
        if isinstance(msg.content, dict):
            return
        if msg.tag.endswith("hypothesis generation"):
            self.hypotheses.append(msg.content)
        elif msg.tag.endswith("ef.feedback"):
            self.hypothesis_decisions[self.hypotheses[-1]] = msg.content.decision
            self.summary_c.markdown(
                "\n".join(
                    (
                        f"{id+1}. :green[{self.hypotheses[id].hypothesis}]\n\t>*{self.hypotheses[id].concise_reason}*"
                        if d
                        else f"{id+1}. {self.hypotheses[id].hypothesis}\n\t>*{self.hypotheses[id].concise_reason}*"
                    )
                    for id, (h, d) in enumerate(self.hypothesis_decisions.items())
                )
            )
        elif msg.tag.endswith("ef.model runner result") or msg.tag.endswith("ef.factor runner result"):
            self.results.append(msg.content.result)
            if len(self.results) == 1:
                self.chart_c.table(self.results[0])
            else:
                df = pd.DataFrame(self.results, index=range(1, len(self.results) + 1))
                fig = px.line(df, x=df.index, y=df.columns, markers=True)
                self.chart_c.plotly_chart(fig)

        self.RDL_win.consume_msg(msg)
        # time.sleep(TIME_DELAY)
