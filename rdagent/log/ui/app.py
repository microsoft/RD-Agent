import argparse
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Type

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from st_btn_select import st_btn_select
from streamlit import session_state as state
from streamlit.delta_generator import DeltaGenerator

from rdagent.components.coder.factor_coder.CoSTEER.evaluators import (
    FactorSingleFeedback,
)
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.components.coder.model_coder.CoSTEER.evaluators import ModelCoderFeedback
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.log.base import Message
from rdagent.log.storage import FileStorage
from rdagent.log.ui.qlib_report_figure import report_figure
from rdagent.scenarios.data_mining.experiment.model_experiment import DMModelScenario
from rdagent.scenarios.general_model.scenario import GeneralModelScenario
from rdagent.scenarios.qlib.experiment.factor_experiment import (
    QlibFactorExperiment,
    QlibFactorScenario,
)
from rdagent.scenarios.qlib.experiment.model_experiment import (
    QlibModelExperiment,
    QlibModelScenario,
)

st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")


# 获取log_path参数
parser = argparse.ArgumentParser(description="RD-Agent Streamlit App")
parser.add_argument("--log_dir", type=str, help="Path to the log directory")
args = parser.parse_args()
if args.log_dir:
    main_log_path = Path(args.log_dir)
    if not main_log_path.exists():
        st.error(f"Log dir `{main_log_path}` does not exist!")
        st.stop()
else:
    main_log_path = None


SELECTED_METRICS = [
    "IC",
    "1day.excess_return_without_cost.annualized_return",
    "1day.excess_return_without_cost.information_ratio",
    "1day.excess_return_without_cost.max_drawdown",
]

if "log_type" not in state:
    state.log_type = "Qlib Model"

if "log_path" not in state:
    if main_log_path:
        state.log_path = next(main_log_path.iterdir()).relative_to(main_log_path)
    else:
        state.log_path = ""

if "fs" not in state:
    state.fs = None

if "msgs" not in state:
    state.msgs = defaultdict(lambda: defaultdict(list))

if "last_msg" not in state:
    state.last_msg = None

if "current_tags" not in state:
    state.current_tags = []

if "lround" not in state:
    state.lround = 0  # RD Loop Round

if "erounds" not in state:
    state.erounds = defaultdict(int)  # Evolving Rounds in each RD Loop

if "e_decisions" not in state:
    state.e_decisions = defaultdict(lambda: defaultdict(tuple))

# Summary Info
if "hypotheses" not in state:
    # Hypotheses in each RD Loop
    state.hypotheses = defaultdict(None)

if "h_decisions" not in state:
    state.h_decisions = defaultdict(bool)

if "metric_series" not in state:
    state.metric_series = []

# Factor Task Baseline
if "alpha158_metrics" not in state:
    state.alpha158_metrics = None


def refresh():
    if main_log_path:
        state.fs = FileStorage(main_log_path / state.log_path).iter_msg()
    else:
        state.fs = FileStorage(state.log_path).iter_msg()
    state.msgs = defaultdict(lambda: defaultdict(list))
    state.lround = 0
    state.erounds = defaultdict(int)
    state.e_decisions = defaultdict(lambda: defaultdict(tuple))
    state.hypotheses = defaultdict(None)
    state.h_decisions = defaultdict(bool)
    state.metric_series = []
    state.last_msg = None
    state.current_tags = []
    state.alpha158_metrics = None


def should_display(msg: Message):
    for t in state.excluded_tags:
        if t in msg.tag.split("."):
            return False

    if type(msg.content).__name__ in state.excluded_types:
        return False

    return True


def get_msgs_until(end_func: Callable[[Message], bool] = lambda _: True):
    if state.fs:
        while True:
            try:
                msg = next(state.fs)
                if should_display(msg):
                    tags = msg.tag.split(".")
                    if "r" not in state.current_tags and "r" in tags:
                        state.lround += 1
                    if "evolving code" not in state.current_tags and "evolving code" in tags:
                        state.erounds[state.lround] += 1

                    state.current_tags = tags
                    state.last_msg = msg

                    # Update Summary Info
                    if "model runner result" in tags or "factor runner result" in tags or "runner result" in tags:
                        # factor baseline exp metrics
                        if state.log_type == "Qlib Factor" and state.alpha158_metrics is None:
                            sms = msg.content.based_experiments[0].result.loc[SELECTED_METRICS]
                            sms.name = "alpha158"
                            state.alpha158_metrics = sms

                        # common metrics
                        if msg.content.result is None:
                            state.metric_series.append(pd.Series([None], index=["AUROC"], name=f"Round {state.lround}"))
                        else:
                            if len(msg.content.result) < 4:
                                ps = msg.content.result
                                ps.index = ["AUROC"]
                                ps.name = f"Round {state.lround}"
                                state.metric_series.append(ps)
                            else:
                                sms = msg.content.result.loc[SELECTED_METRICS]
                                sms.name = f"Round {state.lround}"
                                state.metric_series.append(sms)
                    elif "hypothesis generation" in tags:
                        state.hypotheses[state.lround] = msg.content
                    elif "ef" in tags and "feedback" in tags:
                        state.h_decisions[state.lround] = msg.content.decision
                    elif "d" in tags:
                        if "evolving code" in tags:
                            msg.content = [i for i in msg.content if i]
                        if "evolving feedback" in tags:
                            msg.content = [i for i in msg.content if i]
                            if len(msg.content) != len(state.msgs[state.lround]["d.evolving code"][-1].content):
                                st.toast(":red[**Evolving Feedback Length Error!**]", icon="‼️")
                            right_num = 0
                            for wsf in msg.content:
                                if wsf.final_decision:
                                    right_num += 1
                            wrong_num = len(msg.content) - right_num
                            state.e_decisions[state.lround][state.erounds[state.lround]] = (right_num, wrong_num)

                    state.msgs[state.lround][msg.tag].append(msg)
                    # Stop Getting Logs
                    if end_func(msg):
                        break
            except StopIteration:
                st.toast(":red[**No More Logs to Show!**]", icon="🛑")
                break


def evolving_feedback_window(wsf: FactorSingleFeedback | ModelCoderFeedback):
    if isinstance(wsf, FactorSingleFeedback):
        ffc, efc, cfc, vfc = st.tabs(
            ["**Final Feedback🏁**", "Execution Feedback🖥️", "Code Feedback📄", "Value Feedback🔢"]
        )
        with ffc:
            st.markdown(wsf.final_feedback)
        with efc:
            st.code(wsf.execution_feedback, language="log")
        with cfc:
            st.markdown(wsf.code_feedback)
        with vfc:
            st.markdown(wsf.factor_value_feedback)
    elif isinstance(wsf, ModelCoderFeedback):
        ffc, efc, cfc, msfc, vfc = st.tabs(
            [
                "**Final Feedback🏁**",
                "Execution Feedback🖥️",
                "Code Feedback📄",
                "Model Shape Feedback📐",
                "Value Feedback🔢",
            ]
        )
        with ffc:
            st.markdown(wsf.final_feedback)
        with efc:
            st.code(wsf.execution_feedback, language="log")
        with cfc:
            st.markdown(wsf.code_feedback)
        with msfc:
            st.markdown(wsf.shape_feedback)
        with vfc:
            st.markdown(wsf.value_feedback)


def display_hypotheses(hypotheses: dict[int, Hypothesis], decisions: dict[int, bool], success_only: bool = False):
    if success_only:
        shd = {k: v.__dict__ for k, v in hypotheses.items() if decisions[k]}
    else:
        shd = {k: v.__dict__ for k, v in hypotheses.items()}
    df = pd.DataFrame(shd).T
    if "reason" in df.columns:
        df.drop(["reason"], axis=1, inplace=True)
    df.columns = df.columns.map(lambda x: x.replace("_", " ").capitalize())

    def style_rows(row):
        if decisions[row.name]:
            return ["color: green;"] * len(row)
        return [""] * len(row)

    def style_columns(col):
        if col.name != "Hypothesis":
            return ["font-style: italic;"] * len(col)
        return ["font-weight: bold;"] * len(col)

    # st.dataframe(df.style.apply(style_rows, axis=1).apply(style_columns, axis=0))
    st.markdown(df.style.apply(style_rows, axis=1).apply(style_columns, axis=0).to_html(), unsafe_allow_html=True)


def metrics_window(df: pd.DataFrame, R: int, C: int, *, height: int = 300, colors: list[str] = None):
    fig = make_subplots(rows=R, cols=C, subplot_titles=df.columns)

    def hypothesis_hover_text(h: Hypothesis, d: bool = False):
        color = "green" if d else "black"
        text = h.hypothesis
        lines = textwrap.wrap(text, width=60)
        return f"<span style='color: {color};'>{'<br>'.join(lines)}</span>"

    hover_texts = [
        hypothesis_hover_text(state.hypotheses[int(i[6:])], state.h_decisions[int(i[6:])])
        for i in df.index
        if i != "alpha158"
    ]
    if state.alpha158_metrics is not None:
        hover_texts = ["Baseline: alpha158"] + hover_texts
    for ci, col in enumerate(df.columns):
        row = ci // C + 1
        col_num = ci % C + 1
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=col,
                mode="lines+markers",
                connectgaps=True,
                marker=dict(size=10, color=colors[ci]) if colors else dict(size=10),
                hovertext=hover_texts,
                hovertemplate="%{hovertext}<br><br><span style='color: black'>%{x} Value:</span> <span style='color: blue'>%{y}</span><extra></extra>",
            ),
            row=row,
            col=col_num,
        )
    fig.update_layout(showlegend=False, height=height)

    if state.alpha158_metrics is not None:
        for i in range(1, R + 1):  # 行
            for j in range(1, C + 1):  # 列
                fig.update_xaxes(
                    tickvals=[df.index[0]] + list(df.index[1:]),
                    ticktext=[f'<span style="color:blue; font-weight:bold">{df.index[0]}</span>'] + list(df.index[1:]),
                    row=i,
                    col=j,
                )
    st.plotly_chart(fig)


def summary_window():
    if state.log_type in ["Qlib Model", "Data Mining", "Qlib Factor"]:
        st.header("Summary📊", divider="rainbow", anchor="_summary")
        with st.container():
            # TODO: not fixed height
            with st.container():
                bc, cc = st.columns([2, 2], vertical_alignment="center")
                with bc:
                    st.subheader("Metrics📈", anchor="_metrics")
                with cc:
                    show_true_only = st.toggle("successful hypotheses", value=False)

            # hypotheses_c, chart_c = st.columns([2, 3])
            chart_c = st.container()
            hypotheses_c = st.container()

            with hypotheses_c:
                st.subheader("Hypotheses🏅", anchor="_hypotheses")
                display_hypotheses(state.hypotheses, state.h_decisions, show_true_only)

            with chart_c:
                if state.log_type == "Qlib Factor" and state.alpha158_metrics is not None:
                    df = pd.DataFrame([state.alpha158_metrics] + state.metric_series)
                else:
                    df = pd.DataFrame(state.metric_series)
                if show_true_only and len(state.hypotheses) >= len(state.metric_series):
                    if state.alpha158_metrics is not None:
                        selected = ["alpha158"] + [i for i in df.index if state.h_decisions[int(i[6:])]]
                    else:
                        selected = [i for i in df.index if state.h_decisions[int(i[6:])]]
                    df = df.loc[selected]
                if df.shape[0] == 1:
                    st.table(df.iloc[0])
                elif df.shape[0] > 1:
                    if df.shape[1] == 1:
                        # suhan's scenario
                        fig = px.line(df, x=df.index, y=df.columns, markers=True)
                        fig.update_layout(xaxis_title="Loop Round", yaxis_title=None)
                        st.plotly_chart(fig)
                    else:
                        metrics_window(df, 1, 4, height=300, colors=["red", "blue", "orange", "green"])

    elif state.log_type == "Model from Paper" and len(state.msgs[state.lround]["d.evolving code"]) > 0:
        with st.container(border=True):
            st.subheader("Summary📊", divider="rainbow", anchor="_summary")

            # pass
            ws: list[FactorFBWorkspace | ModelFBWorkspace] = state.msgs[state.lround]["d.evolving code"][-1].content
            # All Tasks

            tab_names = [
                w.target_task.factor_name if isinstance(w.target_task, FactorTask) else w.target_task.name for w in ws
            ]
            for j in range(len(ws)):
                if state.msgs[state.lround]["d.evolving feedback"][-1].content[j].final_decision:
                    tab_names[j] += "✔️"
                else:
                    tab_names[j] += "❌"

            wtabs = st.tabs(tab_names)
            for j, w in enumerate(ws):
                with wtabs[j]:
                    # Evolving Code
                    for k, v in w.code_dict.items():
                        with st.expander(f":green[`{k}`]", expanded=False):
                            st.code(v, language="python")

                    # Evolving Feedback
                    evolving_feedback_window(state.msgs[state.lround]["d.evolving feedback"][-1].content[j])


def tabs_hint():
    st.markdown(
        "<p style='font-size: small; color: #888888;'>You can navigate through the tabs using ⬅️ ➡️ or by holding Shift and scrolling with the mouse wheel🖱️.</p>",
        unsafe_allow_html=True,
    )


# TODO: when tab names are too long, some tabs are not shown
def tasks_window(tasks: list[FactorTask | ModelTask]):
    if isinstance(tasks[0], FactorTask):
        st.markdown("**Factor Tasks🚩**")
        tnames = [f.factor_name for f in tasks]
        if sum(len(tn) for tn in tnames) > 100:
            tabs_hint()
        tabs = st.tabs(tnames)
        for i, ft in enumerate(tasks):
            with tabs[i]:
                # st.markdown(f"**Factor Name**: {ft.factor_name}")
                st.markdown(f"**Description**: {ft.factor_description}")
                st.latex("Formulation")
                st.latex(f"{ft.factor_formulation}")

                mks = "| Variable | Description |\n| --- | --- |\n"
                for v, d in ft.variables.items():
                    mks += f"| ${v}$ | {d} |\n"
                st.markdown(mks)

    elif isinstance(tasks[0], ModelTask):
        st.markdown("**Model Tasks🚩**")
        tnames = [m.name for m in tasks]
        if sum(len(tn) for tn in tnames) > 100:
            tabs_hint()
        tabs = st.tabs(tnames)
        for i, mt in enumerate(tasks):
            with tabs[i]:
                # st.markdown(f"**Model Name**: {mt.name}")
                st.markdown(f"**Model Type**: {mt.model_type}")
                st.markdown(f"**Description**: {mt.description}")
                st.latex("Formulation")
                st.latex(f"{mt.formulation}")

                mks = "| Variable | Description |\n| --- | --- |\n"
                for v, d in mt.variables.items():
                    mks += f"| ${v}$ | {d} |\n"
                st.markdown(mks)


# Config Sidebar
with st.sidebar:
    st.markdown(
        """
# RD-Agent🤖
## [Scenario Description](#_scenario)
## [Summary](#_summary)
- [**Hypotheses**](#_hypotheses)
- [**Metrics**](#_metrics)
## [RD-Loops](#_rdloops)
- [**Research**](#_research)
- [**Development**](#_development)
- [**Feedback**](#_feedback)
"""
    )

    st.selectbox(
        ":green[**Scenario**]", ["Qlib Model", "Data Mining", "Qlib Factor", "Model from Paper"], key="log_type"
    )

    with st.popover(":orange[**Config⚙️**]"):
        with st.container(border=True):
            st.markdown(":blue[**log path**]")
            if main_log_path:
                if st.toggle("Manual Input"):
                    st.text_input("log path", key="log_path", on_change=refresh)
                else:
                    folders = [
                        folder.relative_to(main_log_path) for folder in main_log_path.iterdir() if folder.is_dir()
                    ]
                    st.selectbox(f"Select from `{main_log_path}`", folders, key="log_path", on_change=refresh)
            else:
                st.text_input("log path", key="log_path", on_change=refresh)

        with st.container(border=True):
            st.markdown(":blue[**excluded configs**]")
            st.multiselect("excluded log tags", ["llm_messages"], ["llm_messages"], key="excluded_tags")
            st.multiselect("excluded log types", ["str", "dict", "list"], ["str"], key="excluded_types")

    if st.button("All Loops"):
        if not state.fs:
            refresh()
        get_msgs_until(lambda m: False)

    if st.button("Next Loop"):
        if not state.fs:
            refresh()
        get_msgs_until(lambda m: "ef.feedback" in m.tag)

    if st.button("One Evolving"):
        if not state.fs:
            refresh()
        get_msgs_until(lambda m: "d.evolving feedback" in m.tag)

    if st.button("refresh logs", help="clear all log messages in cache"):
        refresh()
    debug = st.toggle("debug", value=False)

    if debug:
        if st.button("Single Step Run"):
            if not state.fs:
                refresh()
            get_msgs_until()


# Debug Info Window
if debug:
    with st.expander(":red[**Debug Info**]", expanded=True):
        dcol1, dcol2 = st.columns([1, 3])
        with dcol1:
            st.markdown(
                f"**trace type**: {state.log_type}\n\n"
                f"**log path**: {state.log_path}\n\n"
                f"**excluded tags**: {state.excluded_tags}\n\n"
                f"**excluded types**: {state.excluded_types}\n\n"
                f":blue[**message id**]: {sum(sum(len(tmsgs) for tmsgs in rmsgs.values()) for rmsgs in state.msgs.values())}\n\n"
                f":blue[**round**]: {state.lround}\n\n"
                f":blue[**evolving round**]: {state.erounds[state.lround]}\n\n"
            )
        with dcol2:
            if state.last_msg:
                st.write(state.last_msg)
                if isinstance(state.last_msg.content, list):
                    st.write(state.last_msg.content[0])
                elif not isinstance(state.last_msg.content, str):
                    st.write(state.last_msg.content.__dict__)


# Main Window
header_c1, header_c3 = st.columns([1, 6], vertical_alignment="center")
with st.container():
    with header_c1:
        st.image("https://img-prod-cms-rt-microsoft-com.akamaized.net/cms/api/am/imageFileData/RE1Mu3b?ver=5c31")
    with header_c3:
        st.markdown(
            """
        <h1>
            RD-Agent:<br>LLM-based autonomous evolving agents for industrial data-driven R&D
        </h1>
        """,
            unsafe_allow_html=True,
        )

# Project Info
with st.container():
    image_c, scen_c = st.columns([3, 3], vertical_alignment="center")
    with image_c:
        st.image("./docs/_static/flow.png")
    with scen_c:
        st.header("Scenario Description📖", divider="violet", anchor="_scenario")
        # TODO: other scenarios
        if state.log_type == "Qlib Model":
            st.markdown(QlibModelScenario().rich_style_description)
        elif state.log_type == "Data Mining":
            st.markdown(DMModelScenario().rich_style_description)
        elif state.log_type == "Qlib Factor":
            st.markdown(QlibFactorScenario().rich_style_description)
        elif state.log_type == "Model from Paper":
            st.markdown(GeneralModelScenario().rich_style_description)


# Summary Window
summary_window()

# R&D Loops Window
if state.log_type in ["Qlib Model", "Data Mining", "Qlib Factor"]:
    st.header("R&D Loops♾️", divider="rainbow", anchor="_rdloops")

if state.log_type in ["Qlib Model", "Data Mining", "Qlib Factor"]:
    if len(state.msgs) > 1:
        r_options = list(state.msgs.keys())
        if 0 in r_options:
            r_options.remove(0)
        round = st_btn_select(options=r_options, index=state.lround - 1)
    else:
        round = 1
else:
    round = 1


def research_window():
    with st.container(border=True):
        title = "Research🔍" if state.log_type in ["Qlib Model", "Data Mining", "Qlib Factor"] else "Research🔍 (reader)"
        st.subheader(title, divider="blue", anchor="_research")
        if state.log_type in ["Qlib Model", "Data Mining", "Qlib Factor"]:
            # pdf image
            if pim := state.msgs[round]["r.extract_factors_and_implement.load_pdf_screenshot"]:
                for i in range(min(2, len(pim))):
                    st.image(pim[i].content, use_column_width=True)

            # Hypothesis
            if hg := state.msgs[round]["r.hypothesis generation"]:
                st.markdown("**Hypothesis💡**")  # 🧠
                h: Hypothesis = hg[0].content
                st.markdown(
                    f"""
- **Hypothesis**: {h.hypothesis}
- **Reason**: {h.reason}"""
                )

            if eg := state.msgs[round]["r.experiment generation"]:
                tasks_window(eg[0].content)

        elif state.log_type == "Model from Paper":
            # pdf image
            c1, c2 = st.columns([2, 3])
            with c1:
                if pim := state.msgs[round]["r.pdf_image"]:
                    for i in range(len(pim)):
                        st.image(pim[i].content, use_column_width=True)

            # loaded model exp
            with c2:
                if mem := state.msgs[round]["d.load_experiment"]:
                    me: QlibModelExperiment = mem[0].content
                    tasks_window(me.sub_tasks)


def feedback_window():
    if state.log_type in ["Qlib Model", "Data Mining", "Qlib Factor"]:
        with st.container(border=True):
            st.subheader("Feedback📝", divider="orange", anchor="_feedback")
            if fbr := state.msgs[round]["ef.Quantitative Backtesting Chart"]:
                st.markdown("**Returns📈**")
                fig = report_figure(fbr[0].content)
                st.plotly_chart(fig)
            if fb := state.msgs[round]["ef.feedback"]:
                st.markdown("**Hypothesis Feedback🔍**")
                h: HypothesisFeedback = fb[0].content
                st.markdown(
                    f"""
- **Observations**: {h.observations}
- **Hypothesis Evaluation**: {h.hypothesis_evaluation}
- **New Hypothesis**: {h.new_hypothesis}
- **Decision**: {h.decision}
- **Reason**: {h.reason}"""
                )


if state.log_type in ["Qlib Model", "Data Mining", "Qlib Factor"]:
    rf_c, d_c = st.columns([2, 2])
elif state.log_type == "Model from Paper":
    rf_c = st.container()
    d_c = st.container()


with rf_c:
    research_window()
    feedback_window()


# Development Window (Evolving)
with d_c.container(border=True):
    title = (
        "Development🛠️"
        if state.log_type in ["Qlib Model", "Data Mining", "Qlib Factor"]
        else "Development🛠️ (evolving coder)"
    )
    st.subheader(title, divider="green", anchor="_development")

    # Evolving Status
    if state.erounds[round] > 0:
        st.markdown("**☑️ Evolving Status**")
        es = state.e_decisions[round]
        e_status_mks = "".join(f"| {ei} " for ei in range(1, state.erounds[round] + 1)) + "|\n"
        e_status_mks += "|--" * state.erounds[round] + "|\n"
        for ei, estatus in es.items():
            if not estatus:
                estatus = (0, 0)
            e_status_mks += "| " + "✔️<br>" * estatus[0] + "❌<br>" * estatus[1] + " "
        e_status_mks += "|\n"
        st.markdown(e_status_mks, unsafe_allow_html=True)

    # Evolving Tabs
    if state.erounds[round] > 0:
        if state.erounds[round] > 1:
            st.markdown("**🔄️Evolving Rounds**")
            evolving_round = st_btn_select(
                options=range(1, state.erounds[round] + 1), index=state.erounds[round] - 1, key="show_eround"
            )
        else:
            evolving_round = 1

        ws: list[FactorFBWorkspace | ModelFBWorkspace] = state.msgs[round]["d.evolving code"][
            evolving_round - 1
        ].content
        # All Tasks

        tab_names = [
            w.target_task.factor_name if isinstance(w.target_task, FactorTask) else w.target_task.name for w in ws
        ]
        if len(state.msgs[round]["d.evolving feedback"]) >= evolving_round:
            for j in range(len(ws)):
                if state.msgs[round]["d.evolving feedback"][evolving_round - 1].content[j].final_decision:
                    tab_names[j] += "✔️"
                else:
                    tab_names[j] += "❌"
        if sum(len(tn) for tn in tab_names) > 100:
            tabs_hint()
        wtabs = st.tabs(tab_names)
        for j, w in enumerate(ws):
            with wtabs[j]:
                # Evolving Code
                for k, v in w.code_dict.items():
                    with st.expander(f":green[`{k}`]", expanded=True):
                        st.code(v, language="python")

                # Evolving Feedback
                if len(state.msgs[round]["d.evolving feedback"]) >= evolving_round:
                    evolving_feedback_window(state.msgs[round]["d.evolving feedback"][evolving_round - 1].content[j])


with st.container(border=True):
    st.subheader("Disclaimer", divider="gray")
    st.markdown(
        "This content is AI-generated and may not be fully accurate or up-to-date; please verify with a professional for critical matters."
    )
