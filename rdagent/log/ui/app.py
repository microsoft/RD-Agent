import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Type
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
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
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import (
    QlibModelExperiment,
    QlibModelScenario,
)
from rdagent.app.model_extraction_and_code.GeneralModel import GeneralModelScenario
from st_btn_select import st_btn_select

st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")

main_log_path = Path('/data/userdata/share')

if "log_type" not in state:
    state.log_type = "qlib_model"

if "log_path" not in state:
    state.log_path = next(main_log_path.iterdir()).relative_to(main_log_path)

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


def refresh():
    state.fs = FileStorage(main_log_path / state.log_path).iter_msg()
    state.msgs = defaultdict(lambda: defaultdict(list))
    state.lround = 0
    state.erounds = defaultdict(int)
    state.hypotheses = defaultdict(None)
    state.h_decisions = defaultdict(bool)
    state.metric_series = []
    state.last_msg = None
    state.current_tags = []


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
                        if msg.content.result is None:
                            state.metric_series.append(pd.Series([None], index=["AUROC"]))
                        else:
                            if len(msg.content.result) < 4:
                                ps = msg.content.result
                                ps.index = ["AUROC"]
                                state.metric_series.append(ps)
                            else:
                                state.metric_series.append(
                                    msg.content.result.loc[
                                        [
                                            "IC",
                                            "1day.excess_return_without_cost.annualized_return",
                                            "1day.excess_return_without_cost.information_ratio",
                                            "1day.excess_return_without_cost.max_drawdown",
                                        ]
                                    ]
                                )
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
                st.toast(':red[**No More Logs to Show!**]', icon='🛑')
                break


def summary_window():
    if state.log_type in ["qlib_model", "qlib_factor"]:
        with st.container():
            st.header("Summary📊", divider="rainbow", anchor="_summary")
            # TODO: not fixed height
            with st.container():
                ac,bc,cc = st.columns([2,1,2], vertical_alignment="center")
                with ac:
                    st.subheader("Hypotheses🏅", anchor="_hypotheses")
                with bc:
                    st.subheader("Metrics📈", anchor="_metrics")
                with cc:
                    show_true_only = st.toggle("successfully hypothesis", value=False)
            
            hypotheses_c, chart_c = st.columns([2, 3])
            with hypotheses_c:
                with st.container(height=700):
                    h_str = "\n".join(
                        f"{id}. :green[**{h.hypothesis}**]\n\t>:green-background[*{h.__dict__.get('concise_reason', '')}*]"
                        if state.h_decisions[id]
                        else f"{id}. {h.hypothesis}\n\t>*{h.__dict__.get('concise_reason', '')}*"
                        for id, h in state.hypotheses.items()
                    )
                    st.markdown(h_str)
            
            with chart_c:
                with st.container(height=700):
                    labels = [f"Round {i}" for i in range(1, len(state.metric_series) + 1)]
                    df = pd.DataFrame(state.metric_series, index=labels)
                    if show_true_only and len(state.hypotheses) >= len(state.metric_series):
                        df = df.iloc[[i for i in range(df.shape[0]) if state.h_decisions[i + 1]]]
                    if df.shape[0] == 1:
                        st.table(df.iloc[0])
                    elif df.shape[0] > 1:
                        # TODO: figure label
                        # TODO: separate into different figures
                        if df.shape[1] == 1:
                            # suhan's scenario
                            fig = px.line(df, x=df.index, y=df.columns, markers=True)
                            fig.update_layout(xaxis_title="Loop Round", yaxis_title=None)
                        else:
                            # 2*2 figure
                            fig = make_subplots(rows=2, cols=2, subplot_titles=df.columns)
                            colors = ['red', 'blue', 'orange', 'green']
                            for ci, col in enumerate(df.columns):
                                row = ci // 2 + 1
                                col_num = ci % 2 + 1
                                fig.add_trace(
                                    go.Scatter(x=df.index,
                                               y=df[col],
                                               name=col,
                                               mode="lines+markers",
                                               connectgaps=True,
                                               marker=dict(size=10, color=colors[ci]),
                                               ),
                                    row=row,
                                    col=col_num
                                )
                            fig.update_layout(showlegend=False, height=650)
                        st.plotly_chart(fig)


def tasks_window(tasks: list[FactorTask | ModelTask]):
    if isinstance(tasks[0], FactorTask):
        st.markdown("**Factor Tasks**")
        tabs = st.tabs([f.factor_name for f in tasks])
        for i, ft in enumerate(tasks):
            with tabs[i]:
                # st.markdown(f"**Factor Name**: {ft.factor_name}")
                st.markdown(f"**Description**: {ft.factor_description}")
                st.latex("Formulation")
                st.latex(f"{ft.factor_formulation}")

                mks = "| Variable | Description |\n| --- | --- |\n"
                for v,d in ft.variables.items():
                    mks += f"| ${v}$ | {d} |\n"
                st.markdown(mks)

    elif isinstance(tasks[0], ModelTask):
        st.markdown("**Model Tasks**")
        tabs = st.tabs([m.name for m in tasks])
        for i, mt in enumerate(tasks):
            with tabs[i]:
                # st.markdown(f"**Model Name**: {mt.name}")
                st.markdown(f"**Model Type**: {mt.model_type}")
                st.markdown(f"**Description**: {mt.description}")
                st.latex("Formulation")
                st.latex(f"{mt.formulation}")

                mks = "| Variable | Description |\n| --- | --- |\n"
                for v,d in mt.variables.items():
                    mks += f"| ${v}$ | {d} |\n"
                st.markdown(mks)


# Config Sidebar
with st.sidebar:

    with st.popover(":orange[**Config**]"):
        with st.container(border=True):
            st.markdown(":blue[**log path**]")
            if st.toggle("Manual Input"):
                st.text_input("log path", key="log_path", on_change=refresh)
            else:
                folders = [folder.relative_to(main_log_path) for folder in main_log_path.iterdir() if folder.is_dir()]
                st.selectbox("Select from `ep03:/data/userdata/share`", folders, key="log_path", on_change=refresh)

            st.selectbox(":blue[**trace type**]", ["qlib_model", "qlib_factor", "model_extraction_and_implementation"], key="log_type")

        with st.container(border=True):
            st.markdown(":blue[**excluded configs**]")
            st.multiselect("excluded log tags", ["llm_messages"], ["llm_messages"], key="excluded_tags")
            st.multiselect("excluded log types", ["str", "dict", "list"], ["str"], key="excluded_types")
    
    st.markdown("""
# RD-Agent🤖
## [Scenario Description](#_scenario)
## [Summary](#_summary)
- [**Hypotheses**](#_hypotheses)
- [**Metrics**](#_metrics)
## [RD-Loops](#_rdloops)
- [**Research**](#_research)
- [**Development**](#_development)
- [**Feedback**](#_feedback)
""")

    st.divider()

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
                    st.write(state.last_msg.content)


# Main Window

# Project Info
with st.container():
    image_c, scen_c = st.columns([3, 3], vertical_alignment="center")
    with image_c:
        st.image("./docs/_static/flow.png")
    with scen_c:
        st.header("Scenario Description📖", divider="violet", anchor="_scenario")
        # TODO: other scenarios
        if state.log_type == "qlib_model":
            st.markdown(QlibModelScenario().rich_style_description)
        elif state.log_type == "model_extraction_and_implementation":
            st.markdown(GeneralModelScenario().rich_style_description)


# Summary Window
summary_window()

# R&D Loops Window
st.header("R&D Loops♾️", divider="rainbow", anchor="_rdloops")

if len(state.msgs) > 1:
    round = st_btn_select(options=state.msgs.keys(), index=state.lround-1)
else:
    round = 1


def research_window():
    st.subheader("Research🔍", divider="blue", anchor="_research")
    if state.log_type in ["qlib_model", "qlib_factor"]:
        # pdf image
        if pim := state.msgs[round]["r.extract_factors_and_implement.load_pdf_screenshot"]:
            for i in range(min(2, len(pim))):
                st.image(pim[i].content, width=200)

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

    elif state.log_type == "model_extraction_and_implementation":
        # pdf image
        if pim := state.msgs[round]["r.pdf_image"]:
            for i in range(len(pim)):
                st.image(pim[i].content, width=200)

        # loaded model exp
        if mem := state.msgs[round]["d.load_experiment"]:
            me: QlibModelExperiment = mem[0].content
            tasks_window(me.sub_tasks)


def feedback_window():
    st.subheader("Feedback📝", divider="orange", anchor="_feedback")
    if state.log_type in ["qlib_model", "qlib_factor"]:
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
    elif state.log_type == "model_extraction_and_implementation":
        if fbr := state.msgs[round]["d.developed_experiment"]:
            st.markdown("**Returns📈**")
            result_df = fbr[0].content.result
            if result_df:
                fig = report_figure(result_df)
                st.plotly_chart(fig)
            else:
                st.markdown("Returns is None")

# Research & Feedback Window
rf_c, d_c = st.columns([2, 2])
with rf_c:
    with st.container(border=True):
        research_window()

    with st.container(border=True):
        feedback_window()


# Development Window (Evolving)
with d_c.container(border=True):
    st.subheader("Development🛠️", divider="green", anchor="_development")

    # Evolving Status
    if state.erounds[round] > 0:
        st.markdown("**🔸Evolving Status🔸**")
        es = state.e_decisions[round]
        e_status_mks = "".join(f"| {ei} " for ei in range(1, state.erounds[round]+1)) + "|\n"
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
            st.markdown("**🔹Evolving Rounds🔹**")
            evolving_round = st_btn_select(options=range(1, state.erounds[round]+1),index=state.erounds[round]-1)
        else:
            evolving_round = 1

        ws: list[FactorFBWorkspace | ModelFBWorkspace] = state.msgs[round]["d.evolving code"][evolving_round-1].content
        # All Tasks

        tab_names = [w.target_task.factor_name if isinstance(w.target_task, FactorTask) else w.target_task.name for w in ws]
        if len(state.msgs[round]["d.evolving feedback"]) >= evolving_round:
            for j in range(len(ws)):
                if state.msgs[round]["d.evolving feedback"][evolving_round-1].content[j].final_decision:
                    tab_names[j] += "✔️"
                else:
                    tab_names[j] += "❌"

        wtabs = st.tabs(tab_names)
        for j, w in enumerate(ws):
            with wtabs[j]:
                # Evolving Code
                for k, v in w.code_dict.items():
                    with st.expander(f":green[`{k}`]", expanded=True):
                        st.code(v, language="python")

                # Evolving Feedback
                if len(state.msgs[round]["d.evolving feedback"]) >= evolving_round:
                    wsf: list[FactorSingleFeedback | ModelCoderFeedback] = state.msgs[round]["d.evolving feedback"][evolving_round-1].content[j]
                    if isinstance(wsf, FactorSingleFeedback):
                        ffc, efc, cfc, vfc = st.tabs(['**Final Feedback🏁**', 'Execution Feedback🖥️', 'Code Feedback📄', 'Value Feedback🔢'])
                        with ffc:
                            st.markdown(wsf.final_feedback)
                        with efc:
                            st.code(wsf.execution_feedback, language="log")
                        with cfc:
                            st.markdown(wsf.code_feedback)
                        with vfc:
                            st.markdown(wsf.factor_value_feedback)
                    elif isinstance(wsf, ModelCoderFeedback):
                        ffc, efc, cfc, msfc, vfc = st.tabs(['**Final Feedback🏁**', 'Execution Feedback🖥️', 'Code Feedback📄', 'Model Shape Feedback📐', 'Value Feedback🔢'])
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
