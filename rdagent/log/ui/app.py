import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Type

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

st.set_page_config(layout="wide")


if "log_path" not in state:
    state.log_path = ""

if "log_type" not in state:
    state.log_type = "qlib_model"

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

# Summary Info
if "hypotheses" not in state:
    # Hypotheses in each RD Loop
    state.hypotheses = defaultdict(None)

if "h_decisions" not in state:
    state.h_decisions = defaultdict(bool)

if "metric_series" not in state:
    state.metric_series = []


def refresh():
    state.fs = FileStorage(state.log_path).iter_msg()
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
                    state.msgs[state.lround][msg.tag].append(msg)

                    # Update Summary Info
                    if "model runner result" in tags or "factor runner result" in tags or "runner result" in tags:
                        if msg.content.result is None:
                            state.metric_series.append(pd.Series([None], index=["AUROC"]))
                        else:
                            if msg.content.result.name == "AUROC":
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

                    # Stop Getting Logs
                    if end_func(msg):
                        break
            except StopIteration:
                break


# Config Sidebar
with st.sidebar:
    st.text_input("log path", key="log_path", on_change=refresh)
    st.selectbox("trace type", ["qlib_model", "qlib_factor", "model_extraction_and_implementation"], key="log_type")

    st.multiselect("excluded log tags", ["llm_messages"], ["llm_messages"], key="excluded_tags")
    st.multiselect("excluded log types", ["str", "dict", "list"], ["str"], key="excluded_types")

    if st.button("refresh"):
        refresh()
    debug = st.checkbox("debug", value=False)

    if debug:
        if st.button("Single Step Run"):
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
    image_c, toc_c = st.columns([3, 3], vertical_alignment="center")
    with image_c:
        st.image("./docs/_static/scen.jpg")
    with toc_c:
        st.markdown(
            """
# RD-Agent🤖
## [Scenario Description](#_scenario)
## [Summary](#_summary)
## [RD-Loops](#_rdloops)
### [Research](#_research)
### [Development](#_development)
### [Feedback](#_feedback)
"""
        )
with st.container(border=True):
    st.header("Scenario Description📖", divider=True, anchor="_scenario")
    # TODO: other scenarios
    if state.log_type == "qlib_model":
        st.markdown(QlibModelScenario().rich_style_description)
    elif state.log_type == "model_extraction_and_implementation":
        st.markdown(
            """
# General Model Scenario

## Overview

This demo automates the extraction and iterative development of models from academic papers, ensuring functionality and correctness.

### Scenario: Auto-Developing Model Code from Academic Papers

#### Overview

This scenario automates the development of PyTorch models by reading academic papers or other sources. It supports various data types, including tabular, time-series, and graph data. The primary workflow involves two main components: the Reader and the Coder.

#### Workflow Components

1. **Reader**
- Parses and extracts relevant model information from academic papers or sources, including architectures, parameters, and implementation details.
- Uses Large Language Models to convert content into a structured format for the Coder.

2. **Evolving Coder**
- Translates structured information from the Reader into executable PyTorch code.
- Utilizes an evolving coding mechanism to ensure correct tensor shapes, verified with sample input tensors.
- Iteratively refines the code to align with source material specifications.

#### Supported Data Types

- **Tabular Data:** Structured data with rows and columns, such as spreadsheets or databases.
- **Time-Series Data:** Sequential data points indexed in time order, useful for forecasting and temporal pattern recognition.
- **Graph Data:** Data structured as nodes and edges, suitable for network analysis and relational tasks.
"""
        )


# Summary Window
@st.experimental_fragment()
def summary_window():
    if state.log_type in ["qlib_model", "qlib_factor"]:
        with st.container():
            st.header("Summary📊", divider=True, anchor="_summary")
            hypotheses_c, chart_c = st.columns([2, 3])
            # TODO: not fixed height
            with hypotheses_c.container(height=600):
                st.markdown("**Hypotheses🏅**")
                h_str = "\n".join(
                    f"{id}. :green[**{h.hypothesis}**]\n\t>:green-background[*{h.__dict__.get('concise_reason', '')}*]"
                    if state.h_decisions[id]
                    else f"{id}. {h.hypothesis}\n\t>*{h.__dict__.get('concise_reason', '')}*"
                    for id, h in state.hypotheses.items()
                )
                st.markdown(h_str)
            with chart_c.container(height=600):
                mt_c, ms_c = st.columns(2, vertical_alignment="center")
                with mt_c:
                    st.markdown("**Metrics📈**")
                with ms_c:
                    show_true_only = st.checkbox("True Decisions Only", value=False)

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
                        fig.update_layout(legend_title_text="Metrics", xaxis_title="Loop Round", yaxis_title=None)
                    else:
                        # 2*2 figure
                        fig = make_subplots(rows=2, cols=2, subplot_titles=df.columns)
                        for ci, col in enumerate(df.columns):
                            row = ci // 2 + 1
                            col_num = ci % 2 + 1
                            fig.add_trace(
                                go.Scatter(x=df.index, y=df[col], mode="lines+markers", name=col), row=row, col=col_num
                            )
                        fig.update_layout(title_text="Metrics", showlegend=False)
                    st.plotly_chart(fig)


summary_window()

# R&D Loops Window
st.header("R&D Loops♾️", divider=True, anchor="_rdloops")
button_c1, button_c2, round_s_c = st.columns([2, 3, 18], vertical_alignment="center")
with button_c1:
    if st.button("Run One Loop"):
        get_msgs_until(lambda m: "ef.feedback" in m.tag)
with button_c2:
    if st.button("Run One Evolving Step"):
        get_msgs_until(lambda m: "d.evolving feedback" in m.tag)

if len(state.msgs) > 1:
    with round_s_c:
        round = st.select_slider("Select RDLoop Round", options=state.msgs.keys(), value=state.lround)
else:
    round = 1

rf_c, d_c = st.columns([2, 2])

# Research & Feedback Window
with rf_c:
    if state.log_type in ["qlib_model", "qlib_factor"]:
        # Research Window
        with st.container(border=True):
            st.subheader("Research🔍", divider=True, anchor="_research")
            # pdf image
            if pim := state.msgs[round]["r.extract_factors_and_implement.load_pdf_screenshot"]:
                for i in range(min(2, len(pim))):
                    st.image(pim[i].content)

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
                if isinstance(eg[0].content[0], FactorTask):
                    st.markdown("**Factor Tasks**")
                    fts = eg[0].content
                    tabs = st.tabs([f.factor_name for f in fts])
                    for i, ft in enumerate(fts):
                        with tabs[i]:
                            # st.markdown(f"**Factor Name**: {ft.factor_name}")
                            st.markdown(f"**Description**: {ft.factor_description}")
                            st.latex(f"Formulation: {ft.factor_formulation}")

                            variables_df = pd.DataFrame(ft.variables, index=["Description"]).T
                            variables_df.index.name = "Variable"
                            st.table(variables_df)
                elif isinstance(eg[0].content[0], ModelTask):
                    st.markdown("**Model Tasks**")
                    mts = eg[0].content
                    tabs = st.tabs([m.name for m in mts])
                    for i, mt in enumerate(mts):
                        with tabs[i]:
                            # st.markdown(f"**Model Name**: {mt.name}")
                            st.markdown(f"**Model Type**: {mt.model_type}")
                            st.markdown(f"**Description**: {mt.description}")
                            st.latex(f"Formulation: {mt.formulation}")

                            variables_df = pd.DataFrame(mt.variables, index=["Value"]).T
                            variables_df.index.name = "Variable"
                            st.table(variables_df)

        # Feedback Window
        with st.container(border=True):
            st.subheader("Feedback📝", divider=True, anchor="_feedback")
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
        # Research Window
        with st.container(border=True):
            # pdf image
            st.subheader("Research🔍", divider=True, anchor="_research")
            if pim := state.msgs[round]["r.pdf_image"]:
                for i in range(len(pim)):
                    st.image(pim[i].content)

            # loaded model exp
            if mem := state.msgs[round]["d.load_experiment"]:
                me: QlibModelExperiment = mem[0].content
                mts: list[ModelTask] = me.sub_tasks
                tabs = st.tabs([m.name for m in mts])
                for i, mt in enumerate(mts):
                    with tabs[i]:
                        # st.markdown(f"**Model Name**: {mt.name}")
                        st.markdown(f"**Model Type**: {mt.model_type}")
                        st.markdown(f"**Description**: {mt.description}")
                        st.latex(f"Formulation: {mt.formulation}")

                        variables_df = pd.DataFrame(mt.variables, index=["Value"]).T
                        variables_df.index.name = "Variable"
                        st.table(variables_df)

        # Feedback Window
        with st.container(border=True):
            st.subheader("Feedback📝", divider=True, anchor="_feedback")
            if fbr := state.msgs[round]["d.developed_experiment"]:
                st.markdown("**Returns📈**")
                result_df = fbr[0].content.result
                if result_df:
                    fig = report_figure(result_df)
                    st.plotly_chart(fig)
                else:
                    st.markdown("Returns is None")


# Development Window (Evolving)
with d_c.container(border=True):
    st.subheader("Development🛠️", divider=True, anchor="_development")
    # Evolving Tabs
    if state.erounds[round] > 0:
        etabs = st.tabs([str(i) for i in range(1, state.erounds[round] + 1)])

    for i in range(0, state.erounds[round]):
        with etabs[i]:
            ws: list[FactorFBWorkspace | ModelFBWorkspace] = state.msgs[round]["d.evolving code"][i].content
            ws = [w for w in ws if w]
            # All Tasks

            tab_names = [
                w.target_task.factor_name if isinstance(w.target_task, FactorTask) else w.target_task.name for w in ws
            ]
            wtabs = st.tabs(tab_names)
            for j, w in enumerate(ws):
                with wtabs[j]:
                    # Evolving Code
                    for k, v in w.code_dict.items():
                        with st.expander(f":green[`{k}`]", expanded=True):
                            st.code(v, language="python")

                    # Evolving Feedback
                    if len(state.msgs[round]["d.evolving feedback"]) > i:
                        wsf: list[FactorSingleFeedback | ModelCoderFeedback] = state.msgs[round]["d.evolving feedback"][
                            i
                        ].content[j]
                        if isinstance(wsf, FactorSingleFeedback):
                            st.markdown(
                                f"""#### :blue[Factor Execution Feedback]
{wsf.execution_feedback}
#### :blue[Factor Code Feedback]
{wsf.code_feedback}
#### :blue[Factor Value Feedback]
{wsf.factor_value_feedback}
#### :blue[Factor Final Feedback]
{wsf.final_feedback}
#### :blue[Factor Final Decision]
This implementation is {'SUCCESS' if wsf.final_decision else 'FAIL'}.
"""
                            )
                        elif isinstance(wsf, ModelCoderFeedback):
                            st.markdown(
                                f"""#### :blue[Model Execution Feedback]
{wsf.execution_feedback}
#### :blue[Model Shape Feedback]
{wsf.shape_feedback}
#### :blue[Model Value Feedback]
{wsf.value_feedback}
#### :blue[Model Code Feedback]
{wsf.code_feedback}
#### :blue[Model Final Feedback]
{wsf.final_feedback}
#### :blue[Model Final Decision]
This implementation is {'SUCCESS' if wsf.final_decision else 'FAIL'}.
"""
                            )

# TODO: evolving tabs -> slider
# TODO: multi tasks SUCCESS/FAIL
# TODO: evolving progress bar, diff colors
