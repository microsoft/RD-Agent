import argparse
import re
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from importlib.resources import files as rfiles
from pathlib import Path
from typing import Callable, Type

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit import session_state as state
from streamlit_theme import st_theme

from rdagent.components.coder.factor_coder.evaluators import FactorSingleFeedback
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.components.coder.model_coder.evaluators import ModelSingleFeedback
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.core.scenario import Scenario
from rdagent.log.base import Message
from rdagent.log.storage import FileStorage
from rdagent.log.ui.qlib_report_figure import report_figure
from rdagent.scenarios.general_model.scenario import GeneralModelScenario
from rdagent.scenarios.kaggle.experiment.scenario import KGScenario
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
from rdagent.scenarios.qlib.experiment.factor_from_report_experiment import (
    QlibFactorFromReportScenario,
)
from rdagent.scenarios.qlib.experiment.model_experiment import (
    QlibModelExperiment,
    QlibModelScenario,
)
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario

st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")


# 获取log_path参数
parser = argparse.ArgumentParser(description="RD-Agent Streamlit App")
parser.add_argument("--log_dir", type=str, help="Path to the log directory")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()
if args.log_dir:
    main_log_path = Path(args.log_dir)
    if not main_log_path.exists():
        st.error(f"Log dir `{main_log_path}` does not exist!")
        st.stop()
else:
    main_log_path = None


QLIB_SELECTED_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.information_ratio",
    "1day.excess_return_with_cost.max_drawdown",
]

SIMILAR_SCENARIOS = (
    QlibModelScenario,
    QlibFactorScenario,
    QlibFactorFromReportScenario,
    QlibQuantScenario,
    KGScenario,
)


def filter_log_folders(main_log_path):
    """
    The webpage only displays valid folders.
    If the __session__ folder exists in a subfolder of the log folder, it is considered a valid folder,
    otherwise it is considered an invalid folder.
    """
    folders = [
        folder.relative_to(main_log_path)
        for folder in main_log_path.iterdir()
        if folder.is_dir() and folder.joinpath("__session__").exists() and folder.joinpath("__session__").is_dir()
    ]
    folders = sorted(folders, key=lambda x: x.name)
    return folders


if "log_path" not in state:
    if main_log_path:
        state.log_path = filter_log_folders(main_log_path)[0]
    else:
        state.log_path = None
        st.toast(":red[**Please Set Log Path!**]", icon="⚠️")

if "scenario" not in state:
    state.scenario = None

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

if "all_metric_series" not in state:
    state.all_metric_series = []

# Factor Task Baseline
if "alpha_baseline_metrics" not in state:
    state.alpha_baseline_metrics = None


def should_display(msg: Message):
    for t in state.excluded_tags + ["debug_tpl", "debug_llm"]:
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
                    if "hypothesis generation" in msg.tag:
                        state.lround += 1

                    # new scenario gen this tags, old version UI not have these tags.
                    msg.tag = re.sub(r"\.evo_loop_\d+", "", msg.tag)
                    msg.tag = re.sub(r"Loop_\d+\.[^.]+", "", msg.tag)
                    msg.tag = re.sub(r"\.\.", ".", msg.tag)

                    # remove old redundant tags
                    msg.tag = re.sub(r"init\.", "", msg.tag)
                    msg.tag = re.sub(r"r\.", "", msg.tag)
                    msg.tag = re.sub(r"d\.", "", msg.tag)
                    msg.tag = re.sub(r"ef\.", "", msg.tag)

                    msg.tag = msg.tag.strip(".")

                    if "evolving code" not in state.current_tags and "evolving code" in tags:
                        state.erounds[state.lround] += 1

                    state.current_tags = tags
                    state.last_msg = msg

                    # Update Summary Info
                    if "runner result" in tags:
                        # factor baseline exp metrics
                        if (
                            isinstance(state.scenario, (QlibFactorScenario, QlibQuantScenario))
                            and state.alpha_baseline_metrics is None
                        ):
                            try:
                                sms = msg.content.based_experiments[0].result
                            except AttributeError:
                                sms = msg.content.based_experiments[0].__dict__["result"]
                            sms = sms.loc[QLIB_SELECTED_METRICS]
                            sms.name = "Alpha Base"
                            state.alpha_baseline_metrics = sms

                        if state.lround == 1 and len(msg.content.based_experiments) > 0:
                            try:
                                sms = msg.content.based_experiments[-1].result
                            except AttributeError:
                                sms = msg.content.based_experiments[-1].__dict__["result"]
                            if sms is not None:
                                if isinstance(
                                    state.scenario,
                                    (
                                        QlibModelScenario,
                                        QlibFactorFromReportScenario,
                                        QlibFactorScenario,
                                        QlibQuantScenario,
                                    ),
                                ):
                                    sms_all = sms
                                    sms = sms.loc[QLIB_SELECTED_METRICS]
                                sms.name = f"Baseline"
                                state.metric_series.append(sms)
                                state.all_metric_series.append(sms_all)

                        # common metrics
                        try:
                            sms = msg.content.result
                        except AttributeError:
                            sms = msg.content.__dict__["result"]
                        if isinstance(
                            state.scenario,
                            (
                                QlibModelScenario,
                                QlibFactorFromReportScenario,
                                QlibFactorScenario,
                                QlibQuantScenario,
                            ),
                        ):
                            sms_all = sms
                            sms = sms.loc[QLIB_SELECTED_METRICS]

                        sms.name = f"Round {state.lround}"
                        sms_all.name = f"Round {state.lround}"
                        state.metric_series.append(sms)
                        state.all_metric_series.append(sms_all)
                    elif "hypothesis generation" in tags:
                        state.hypotheses[state.lround] = msg.content
                    elif "evolving code" in tags:
                        msg.content = [i for i in msg.content if i]
                    elif "evolving feedback" in tags:
                        total_len = len(msg.content)
                        none_num = total_len - len(msg.content)
                        right_num = 0
                        for wsf in msg.content:
                            if wsf.final_decision:
                                right_num += 1
                        wrong_num = len(msg.content) - right_num
                        state.e_decisions[state.lround][state.erounds[state.lround]] = (
                            right_num,
                            wrong_num,
                            none_num,
                        )
                    elif "feedback" in tags and isinstance(msg.content, HypothesisFeedback):
                        state.h_decisions[state.lround] = msg.content.decision

                    state.msgs[state.lround][msg.tag].append(msg)

                    # Stop Getting Logs
                    if end_func(msg):
                        break
            except StopIteration:
                st.toast(":red[**No More Logs to Show!**]", icon="🛑")
                break


def refresh(same_trace: bool = False):
    if state.log_path is None:
        st.toast(":red[**Please Set Log Path!**]", icon="⚠️")
        return

    if main_log_path:
        state.fs = FileStorage(main_log_path / state.log_path).iter_msg()
    else:
        state.fs = FileStorage(state.log_path).iter_msg()

    # detect scenario
    if not same_trace:
        get_msgs_until(lambda m: isinstance(m.content, Scenario))
        if state.last_msg is None or not isinstance(state.last_msg.content, Scenario):
            st.write(state.msgs)
            st.toast(":red[**No Scenario Info detected**]", icon="❗")
            state.scenario = None
        else:
            state.scenario = state.last_msg.content
            st.toast(f":green[**Scenario Info detected**] *{type(state.scenario).__name__}*", icon="✅")

    state.msgs = defaultdict(lambda: defaultdict(list))
    state.lround = 0
    state.erounds = defaultdict(int)
    state.e_decisions = defaultdict(lambda: defaultdict(tuple))
    state.hypotheses = defaultdict(None)
    state.h_decisions = defaultdict(bool)
    state.metric_series = []
    state.all_metric_series = []
    state.last_msg = None
    state.current_tags = []
    state.alpha_baseline_metrics = None


def evolving_feedback_window(wsf: FactorSingleFeedback | ModelSingleFeedback):
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
            st.markdown(wsf.value_feedback)
    elif isinstance(wsf, ModelSingleFeedback):
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
    name_dict = {
        "hypothesis": "RD-Agent proposes the hypothesis⬇️",
        "concise_justification": "because the reason⬇️",
        "concise_observation": "based on the observation⬇️",
        "concise_knowledge": "Knowledge⬇️ gained after practice",
    }
    if success_only:
        shd = {k: v.__dict__ for k, v in hypotheses.items() if decisions[k]}
    else:
        shd = {k: v.__dict__ for k, v in hypotheses.items()}
    df = pd.DataFrame(shd).T

    if "concise_observation" in df.columns and "concise_justification" in df.columns:
        df["concise_observation"], df["concise_justification"] = df["concise_justification"], df["concise_observation"]
        df.rename(
            columns={"concise_observation": "concise_justification", "concise_justification": "concise_observation"},
            inplace=True,
        )
    if "reason" in df.columns:
        df.drop(["reason"], axis=1, inplace=True)
    if "concise_reason" in df.columns:
        df.drop(["concise_reason"], axis=1, inplace=True)

    df.columns = df.columns.map(lambda x: name_dict.get(x, x))
    for col in list(df.columns):
        if all([value is None for value in df[col]]):
            df.drop([col], axis=1, inplace=True)

    def style_rows(row):
        if decisions[row.name]:
            return ["color: green;"] * len(row)
        return [""] * len(row)

    def style_columns(col):
        if col.name != name_dict.get("hypothesis", "hypothesis"):
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
        if i != "Alpha Base" and i != "Baseline"
    ]
    if state.alpha_baseline_metrics is not None:
        hover_texts = ["Baseline"] + hover_texts
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

    if state.alpha_baseline_metrics is not None:
        for i in range(1, R + 1):  # 行
            for j in range(1, C + 1):  # 列
                fig.update_xaxes(
                    tickvals=[df.index[0]] + list(df.index[1:]),
                    ticktext=[f'<span style="color:blue; font-weight:bold">{df.index[0]}</span>'] + list(df.index[1:]),
                    row=i,
                    col=j,
                )
    st.plotly_chart(fig)

    from io import BytesIO

    buffer = BytesIO()
    df.to_csv(buffer)
    buffer.seek(0)
    st.download_button(label="download the metrics (csv)", data=buffer, file_name="metrics.csv", mime="text/csv")


def summary_window():
    if isinstance(state.scenario, SIMILAR_SCENARIOS):
        st.header("Summary📊", divider="rainbow", anchor="_summary")
        if state.lround == 0:
            return
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
                if isinstance(state.scenario, QlibFactorScenario) and state.alpha_baseline_metrics is not None:
                    df = pd.DataFrame([state.alpha_baseline_metrics] + state.metric_series[1:])
                elif isinstance(state.scenario, QlibQuantScenario) and state.alpha_baseline_metrics is not None:
                    df = pd.DataFrame([state.alpha_baseline_metrics] + state.metric_series[1:])
                else:
                    df = pd.DataFrame(state.metric_series)
                if show_true_only and len(state.hypotheses) >= len(state.metric_series):
                    if state.alpha_baseline_metrics is not None:
                        selected = ["Alpha Base"] + [
                            i for i in df.index if i == "Baseline" or state.h_decisions[int(i[6:])]
                        ]
                    else:
                        selected = [i for i in df.index if i == "Baseline" or state.h_decisions[int(i[6:])]]
                    df = df.loc[selected]
                if df.shape[0] == 1:
                    st.table(df.iloc[0])
                elif df.shape[0] > 1:
                    if df.shape[1] == 1:
                        fig = px.line(df, x=df.index, y=df.columns, markers=True)
                        fig.update_layout(xaxis_title="Loop Round", yaxis_title=None)
                        st.plotly_chart(fig)
                    else:
                        metrics_window(df, 1, 4, height=300, colors=["red", "blue", "orange", "green"])

    elif isinstance(state.scenario, GeneralModelScenario):
        with st.container(border=True):
            st.subheader("Summary📊", divider="rainbow", anchor="_summary")
            if len(state.msgs[state.lround]["evolving code"]) > 0:
                # pass
                ws: list[FactorFBWorkspace | ModelFBWorkspace] = state.msgs[state.lround]["evolving code"][-1].content
                # All Tasks

                tab_names = [
                    w.target_task.factor_name if isinstance(w.target_task, FactorTask) else w.target_task.name
                    for w in ws
                ]
                for j in range(len(ws)):
                    if state.msgs[state.lround]["evolving feedback"][-1].content[j].final_decision:
                        tab_names[j] += "✔️"
                    else:
                        tab_names[j] += "❌"

                wtabs = st.tabs(tab_names)
                for j, w in enumerate(ws):
                    with wtabs[j]:
                        # Evolving Code
                        for k, v in w.file_dict.items():
                            with st.expander(f":green[`{k}`]", expanded=False):
                                st.code(v, language="python")

                        # Evolving Feedback
                        evolving_feedback_window(state.msgs[state.lround]["evolving feedback"][-1].content[j])


def tabs_hint():
    st.markdown(
        "<p style='font-size: small; color: #888888;'>You can navigate through the tabs using ⬅️ ➡️ or by holding Shift and scrolling with the mouse wheel🖱️.</p>",
        unsafe_allow_html=True,
    )


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
                st.latex(ft.factor_formulation)

                mks = "| Variable | Description |\n| --- | --- |\n"
                if isinstance(ft.variables, dict):
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
                st.latex(mt.formulation)

                mks = "| Variable | Description |\n| --- | --- |\n"
                if mt.variables:
                    for v, d in mt.variables.items():
                        mks += f"| ${v}$ | {d} |\n"
                    st.markdown(mks)
                st.markdown(f"**Train Para**: {mt.training_hyperparameters}")


def research_window():
    with st.container(border=True):
        title = "Research🔍" if isinstance(state.scenario, SIMILAR_SCENARIOS) else "Research🔍 (reader)"
        st.subheader(title, divider="blue", anchor="_research")
        if isinstance(state.scenario, SIMILAR_SCENARIOS):
            # pdf image
            if pim := state.msgs[round]["load_pdf_screenshot"]:
                for i in range(min(2, len(pim))):
                    st.image(pim[i].content, use_container_width=True)

            # Hypothesis
            if hg := state.msgs[round]["hypothesis generation"]:
                st.markdown("**Hypothesis💡**")  # 🧠
                h: Hypothesis = hg[0].content
                st.markdown(
                    f"""
- **Hypothesis**: {h.hypothesis}
- **Reason**: {h.reason}"""
                )

            if eg := state.msgs[round]["experiment generation"]:
                tasks_window(eg[0].content)

        elif isinstance(state.scenario, GeneralModelScenario):
            # pdf image
            c1, c2 = st.columns([2, 3])
            with c1:
                if pim := state.msgs[0]["pdf_image"]:
                    for i in range(len(pim)):
                        st.image(pim[i].content, use_container_width=True)

            # loaded model exp
            with c2:
                if mem := state.msgs[0]["load_experiment"]:
                    me: QlibModelExperiment = mem[0].content
                    tasks_window(me.sub_tasks)


def feedback_window():
    # st.write(round)
    # # Check if metric series exists and has the matching round
    # if state.all_metric_series:
    #     for metric in state.all_metric_series:
    #         if metric.name == f"Round {round}":
    #             # Select specific metrics with cost
    #             selected_metrics_with_cost = {
    #                 'IC': float(f"{metric['IC']:.4f}"),
    #                 'ICIR': float(f"{metric['ICIR']:.4f}"),
    #                 'Rank IC': float(f"{metric['Rank IC']:.4f}"),
    #                 'Rank ICIR': float(f"{metric['Rank ICIR']:.4f}"),
    #                 'ARR': float(f"{metric['1day.excess_return_with_cost.annualized_return']:.4f}"),
    #                 'IR': float(f"{metric['1day.excess_return_with_cost.information_ratio']:.4f}"),
    #                 'MDD': float(f"{metric['1day.excess_return_with_cost.max_drawdown']:.4f}"),
    #                 'Sharpe': float(f"{metric['1day.excess_return_with_cost.annualized_return'] / abs(metric['1day.excess_return_with_cost.max_drawdown']):.4f}")
    #             }
    #             st.write("With Cost Metrics:")
    #             st.write(pd.Series(selected_metrics_with_cost))

    #             # Select specific metrics without cost
    #             selected_metrics_without_cost = {
    #                 'IC': float(f"{metric['IC']:.4f}"),
    #                 'ICIR': float(f"{metric['ICIR']:.4f}"),
    #                 'Rank IC': float(f"{metric['Rank IC']:.4f}"),
    #                 'Rank ICIR': float(f"{metric['Rank ICIR']:.4f}"),
    #                 'ARR': float(f"{metric['1day.excess_return_without_cost.annualized_return']:.4f}"),
    #                 'IR': float(f"{metric['1day.excess_return_without_cost.information_ratio']:.4f}"),
    #                 'MDD': float(f"{metric['1day.excess_return_without_cost.max_drawdown']:.4f}"),
    #                 'Sharpe': float(f"{metric['1day.excess_return_without_cost.annualized_return'] / abs(metric['1day.excess_return_without_cost.max_drawdown']):.4f}")
    #             }
    #             st.write("Without Cost Metrics:")
    #             st.write(pd.Series(selected_metrics_without_cost))
    #             break
    if isinstance(state.scenario, SIMILAR_SCENARIOS):
        with st.container(border=True):
            st.subheader("Feedback📝", divider="orange", anchor="_feedback")

            if state.lround > 0 and isinstance(
                state.scenario,
                (QlibModelScenario, QlibFactorScenario, QlibFactorFromReportScenario, QlibQuantScenario, KGScenario),
            ):
                if fbr := state.msgs[round]["runner result"]:
                    try:
                        st.write("workspace")
                        st.write(fbr[0].content.experiment_workspace.workspace_path)
                        st.write(fbr[0].content.stdout)
                    except Exception as e:
                        st.error(f"Error displaying workspace path: {str(e)}")
                with st.expander("**Config⚙️**", expanded=True):
                    st.markdown(state.scenario.experiment_setting, unsafe_allow_html=True)

            if fbr := state.msgs[round]["Quantitative Backtesting Chart"]:
                st.markdown("**Returns📈**")
                fig = report_figure(fbr[0].content)
                st.plotly_chart(fig)
            if fb := state.msgs[round]["feedback"]:
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

            if isinstance(state.scenario, KGScenario):
                if fbe := state.msgs[round]["runner result"]:
                    submission_path = fbe[0].content.experiment_workspace.workspace_path / "submission.csv"
                    st.markdown(
                        f":green[**Exp Workspace**]: {str(fbe[0].content.experiment_workspace.workspace_path.absolute())}"
                    )
                    try:
                        data = submission_path.read_bytes()
                        st.download_button(
                            label="**Download** submission.csv",
                            data=data,
                            file_name="submission.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.markdown(f":red[**Download Button Error**]: {e}")


def evolving_window():
    title = "Development🛠️" if isinstance(state.scenario, SIMILAR_SCENARIOS) else "Development🛠️ (evolving coder)"
    st.subheader(title, divider="green", anchor="_development")

    # Evolving Status
    if state.erounds[round] > 0:
        st.markdown("**☑️ Evolving Status**")
        es = state.e_decisions[round]
        e_status_mks = "".join(f"| {ei} " for ei in range(1, state.erounds[round] + 1)) + "|\n"
        e_status_mks += "|--" * state.erounds[round] + "|\n"
        for ei, estatus in es.items():
            if not estatus:
                estatus = (0, 0, 0)
            e_status_mks += "| " + "🕙<br>" * estatus[2] + "✔️<br>" * estatus[0] + "❌<br>" * estatus[1] + " "
        e_status_mks += "|\n"
        st.markdown(e_status_mks, unsafe_allow_html=True)

    # Evolving Tabs
    if state.erounds[round] > 0:
        if state.erounds[round] > 1:
            evolving_round = st.radio(
                "**🔄️Evolving Rounds**",
                horizontal=True,
                options=range(1, state.erounds[round] + 1),
                index=state.erounds[round] - 1,
                key="show_eround",
            )
        else:
            evolving_round = 1

        ws: list[FactorFBWorkspace | ModelFBWorkspace] = state.msgs[round]["evolving code"][evolving_round - 1].content
        # All Tasks

        tab_names = [
            w.target_task.factor_name if isinstance(w.target_task, FactorTask) else w.target_task.name for w in ws
        ]
        if len(state.msgs[round]["evolving feedback"]) >= evolving_round:
            for j in range(len(ws)):
                if state.msgs[round]["evolving feedback"][evolving_round - 1].content[j].final_decision:
                    tab_names[j] += "✔️"
                else:
                    tab_names[j] += "❌"
        if sum(len(tn) for tn in tab_names) > 100:
            tabs_hint()
        wtabs = st.tabs(tab_names)
        for j, w in enumerate(ws):
            with wtabs[j]:
                # Evolving Code
                st.markdown(f"**Workspace Path**: {w.workspace_path}")
                for k, v in w.file_dict.items():
                    with st.expander(f":green[`{k}`]", expanded=True):
                        st.code(v, language="python")

                # Evolving Feedback
                if len(state.msgs[round]["evolving feedback"]) >= evolving_round:
                    evolving_feedback_window(state.msgs[round]["evolving feedback"][evolving_round - 1].content[j])


toc = """
## [Scenario Description📖](#_scenario)
## [Summary📊](#_summary)
- [**Metrics📈**](#_metrics)
- [**Hypotheses🏅**](#_hypotheses)
## [RD-Loops♾️](#_rdloops)
- [**Research🔍**](#_research)
- [**Development🛠️**](#_development)
- [**Feedback📝**](#_feedback)
"""
if isinstance(state.scenario, GeneralModelScenario):
    toc = """
## [Scenario Description📖](#_scenario)
### [Summary📊](#_summary)
### [Research🔍](#_research)
### [Development🛠️](#_development)
"""
# Config Sidebar
with st.sidebar:
    st.markdown("# RD-Agent🤖  [:grey[@GitHub]](https://github.com/microsoft/RD-Agent)")
    st.subheader(":blue[Table of Content]", divider="blue")
    st.markdown(toc)
    st.subheader(":orange[Control Panel]", divider="red")

    with st.container(border=True):
        if main_log_path:
            lc1, lc2 = st.columns([1, 2], vertical_alignment="center")
            with lc1:
                st.markdown(":blue[**Log Path**]")
            with lc2:
                manually = st.toggle("Manual Input")
            if manually:
                st.text_input("log path", key="log_path", on_change=refresh, label_visibility="collapsed")
            else:
                folders = filter_log_folders(main_log_path)
                st.selectbox(f"**Select from `{main_log_path}`**", folders, key="log_path", on_change=refresh)
        else:
            st.text_input(":blue[**log path**]", key="log_path", on_change=refresh)

    c1, c2 = st.columns([1, 1], vertical_alignment="center")
    with c1:
        if st.button(":green[**All Loops**]", use_container_width=True):
            if not state.fs:
                refresh()
            get_msgs_until(lambda m: False)
        if st.button("**Reset**", use_container_width=True):
            refresh(same_trace=True)
    with c2:
        if st.button(":green[Next Loop]", use_container_width=True):
            if not state.fs:
                refresh()
            get_msgs_until(lambda m: "feedback" in m.tag and "evolving feedback" not in m.tag)

        if st.button("Next Step", use_container_width=True):
            if not state.fs:
                refresh()
            get_msgs_until(lambda m: "evolving feedback" in m.tag)

    with st.popover(":orange[**Config⚙️**]", use_container_width=True):
        st.multiselect("excluded log tags", ["llm_messages"], ["llm_messages"], key="excluded_tags")
        st.multiselect("excluded log types", ["str", "dict", "list"], ["str"], key="excluded_types")

    if args.debug:
        debug = st.toggle("debug", value=False)

        if debug:
            if st.button("Single Step Run", use_container_width=True):
                get_msgs_until()
    else:
        debug = False


# Debug Info Window
if debug:
    with st.expander(":red[**Debug Info**]", expanded=True):
        dcol1, dcol2 = st.columns([1, 3])
        with dcol1:
            st.markdown(
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
                elif isinstance(state.last_msg.content, dict):
                    st.write(state.last_msg.content)
                elif not isinstance(state.last_msg.content, str):
                    try:
                        st.write(state.last_msg.content.__dict__)
                    except:
                        st.write(type(state.last_msg.content))

if state.log_path and state.fs is None:
    refresh()

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
        img_path = rfiles("rdagent.log.ui").joinpath("flow.png")
        st.image(str(img_path), use_container_width=True)
    with scen_c:
        st.header("Scenario Description📖", divider="violet", anchor="_scenario")
        if state.scenario is not None:
            theme = st_theme()
            if theme:
                theme = theme.get("base", "light")
            css = f"""
<style>
    a[href="#_rdloops"], a[href="#_research"], a[href="#_development"], a[href="#_feedback"], a[href="#_scenario"], a[href="#_summary"], a[href="#_hypotheses"], a[href="#_metrics"] {{
        color: {"black" if theme == "light" else "white"};
    }}
</style>
"""
            st.markdown(state.scenario.rich_style_description + css, unsafe_allow_html=True)


def analyze_task_completion():
    st.header("Task Completion Analysis", divider="orange")

    # Dictionary to store results for all loops
    completion_stats = {}

    # Iterate through all loops
    for loop_round in state.msgs.keys():
        if loop_round == 0:  # Skip initialization round
            continue

        max_evolving_round = state.erounds[loop_round]
        if max_evolving_round == 0:
            continue

        # Track tasks that pass in each evolving round
        tasks_passed_by_round = {}
        cumulative_passed = set()

        # For each evolving round in this loop
        for e_round in range(1, max_evolving_round + 1):
            if len(state.msgs[loop_round]["evolving feedback"]) >= e_round:
                # Get feedback for this evolving round
                feedback = state.msgs[loop_round]["evolving feedback"][e_round - 1].content

                # Count passed tasks and track their indices
                passed_tasks = set()
                for j, task_feedback in enumerate(feedback):
                    if task_feedback.final_decision:
                        passed_tasks.add(j)
                        cumulative_passed.add(j)

                # Store both individual round results and cumulative results
                tasks_passed_by_round[e_round] = {
                    "count": len(passed_tasks),
                    "indices": passed_tasks,
                    "cumulative_count": len(cumulative_passed),
                    "cumulative_indices": cumulative_passed.copy(),
                }

        completion_stats[loop_round] = {
            "total_tasks": len(state.msgs[loop_round]["evolving feedback"][0].content),
            "rounds": tasks_passed_by_round,
            "max_round": max_evolving_round,
        }

    # Display results
    if completion_stats:
        # Add an aggregate view at the top
        st.subheader("🔄 Aggregate Completion Across All Loops")

        # Create summary data for comparison
        summary_data = []
        total_tasks_across_loops = 0
        total_passed_r1 = 0
        total_passed_r3 = 0
        total_passed_r5 = 0
        total_passed_r10 = 0
        total_passed_final = 0

        for loop_round, stats in completion_stats.items():
            total_tasks = stats["total_tasks"]
            total_tasks_across_loops += total_tasks

            # Find data for specific rounds
            r1_passed = stats["rounds"].get(1, {}).get("cumulative_count", 0)
            total_passed_r1 += r1_passed

            # For round 3, use the closest round if exactly 3 doesn't exist
            if 3 in stats["rounds"]:
                r3_passed = stats["rounds"][3]["cumulative_count"]
            elif stats["max_round"] >= 3:
                max_r_below_3 = max([r for r in stats["rounds"].keys() if r <= 3])
                r3_passed = stats["rounds"][max_r_below_3]["cumulative_count"]
            else:
                r3_passed = stats["rounds"][stats["max_round"]]["cumulative_count"] if stats["rounds"] else 0
            total_passed_r3 += r3_passed

            # For round 5, use the closest round if exactly 5 doesn't exist
            if 5 in stats["rounds"]:
                r5_passed = stats["rounds"][5]["cumulative_count"]
            elif stats["max_round"] >= 5:
                max_r_below_5 = max([r for r in stats["rounds"].keys() if r <= 5])
                r5_passed = stats["rounds"][max_r_below_5]["cumulative_count"]
            else:
                r5_passed = stats["rounds"][stats["max_round"]]["cumulative_count"] if stats["rounds"] else 0
            total_passed_r5 += r5_passed

            # For round 10
            if 10 in stats["rounds"]:
                r10_passed = stats["rounds"][10]["cumulative_count"]
            else:
                r10_passed = stats["rounds"][stats["max_round"]]["cumulative_count"] if stats["rounds"] else 0
            total_passed_r10 += r10_passed

            # Final round completion
            final_passed = stats["rounds"][stats["max_round"]]["cumulative_count"] if stats["rounds"] else 0
            total_passed_final += final_passed

            # Add to summary table
            summary_data.append(
                {
                    "Loop": f"Loop {loop_round}",
                    "Total Tasks": total_tasks,
                    "Passed (Round 1)": (
                        f"{r1_passed}/{total_tasks} ({r1_passed/total_tasks:.0%})" if total_tasks > 0 else "N/A"
                    ),
                    "Passed (Round 3)": (
                        f"{r3_passed}/{total_tasks} ({r3_passed/total_tasks:.0%})" if total_tasks > 0 else "N/A"
                    ),
                    "Passed (Round 5)": (
                        f"{r5_passed}/{total_tasks} ({r5_passed/total_tasks:.0%})" if total_tasks > 0 else "N/A"
                    ),
                    "Passed (Final)": (
                        f"{final_passed}/{total_tasks} ({final_passed/total_tasks:.0%})" if total_tasks > 0 else "N/A"
                    ),
                }
            )

        if total_tasks_across_loops > 0:
            summary_data.append(
                {
                    "Loop": "**TOTAL**",
                    "Total Tasks": total_tasks_across_loops,
                    "Passed (Round 1)": f"**{total_passed_r1}/{total_tasks_across_loops} ({total_passed_r1/total_tasks_across_loops:.0%})**",
                    "Passed (Round 3)": f"**{total_passed_r3}/{total_tasks_across_loops} ({total_passed_r3/total_tasks_across_loops:.0%})**",
                    "Passed (Round 5)": f"**{total_passed_r5}/{total_tasks_across_loops} ({total_passed_r5/total_tasks_across_loops:.0%})**",
                    "Passed (Final)": f"**{total_passed_final}/{total_tasks_across_loops} ({total_passed_final/total_tasks_across_loops:.0%})**",
                }
            )

        st.table(pd.DataFrame(summary_data))

        # Summary statistics
        st.markdown("### 📊 Overall Completion Progress:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="After Round 1",
                value=f"{total_passed_r1/total_tasks_across_loops:.0%}",
                help=f"{total_passed_r1}/{total_tasks_across_loops} tasks",
            )
        with col2:
            st.metric(
                label="After Round 3",
                value=f"{total_passed_r3/total_tasks_across_loops:.0%}",
                delta=f"{(total_passed_r3-total_passed_r1)/total_tasks_across_loops:.0%}",
                help=f"{total_passed_r3}/{total_tasks_across_loops} tasks",
            )
        with col3:
            st.metric(
                label="After Round 5",
                value=f"{total_passed_r5/total_tasks_across_loops:.0%}",
                delta=f"{(total_passed_r5-total_passed_r3)/total_tasks_across_loops:.0%}",
                help=f"{total_passed_r5}/{total_tasks_across_loops} tasks",
            )
        with col4:
            st.metric(
                label="Final Completion",
                value=f"{total_passed_final/total_tasks_across_loops:.0%}",
                delta=f"{(total_passed_final-total_passed_r5)/total_tasks_across_loops:.0%}",
                help=f"{total_passed_final}/{total_tasks_across_loops} tasks",
            )

        # Show detailed results by loop
        st.markdown("---")
        st.subheader("Detailed Results by Loop")

        for loop_round, stats in completion_stats.items():
            with st.expander(f"Loop {loop_round} Details"):
                total_tasks = stats["total_tasks"]

                # Create a results table
                data = []
                for e_round in range(1, min(11, stats["max_round"] + 1)):
                    if e_round in stats["rounds"]:
                        round_data = stats["rounds"][e_round]
                        data.append(
                            {
                                "Evolving Round": e_round,
                                "Tasks Passed": f"{round_data['count']}/{total_tasks} ({round_data['count']/total_tasks:.0%})",
                                "Cumulative Passed": f"{round_data['cumulative_count']}/{total_tasks} ({round_data['cumulative_count']/total_tasks:.0%})",
                            }
                        )
                    else:
                        data.append({"Evolving Round": e_round, "Tasks Passed": "N/A", "Cumulative Passed": "N/A"})

                df = pd.DataFrame(data)
                st.table(df)

                st.markdown("### Summary:")
                if 1 in stats["rounds"]:
                    st.markdown(
                        f"- After round 1: **{stats['rounds'][1]['cumulative_count']}/{total_tasks}** tasks passed ({stats['rounds'][1]['cumulative_count']/total_tasks:.0%})"
                    )

                if 3 in stats["rounds"]:
                    st.markdown(
                        f"- After round 3: **{stats['rounds'][3]['cumulative_count']}/{total_tasks}** tasks passed ({stats['rounds'][3]['cumulative_count']/total_tasks:.0%})"
                    )
                elif stats["max_round"] >= 3:
                    max_round_below_3 = max([r for r in stats["rounds"].keys() if r <= 3])
                    st.markdown(
                        f"- After round 3: **{stats['rounds'][max_round_below_3]['cumulative_count']}/{total_tasks}** tasks passed ({stats['rounds'][max_round_below_3]['cumulative_count']/total_tasks:.0%})"
                    )

                if 5 in stats["rounds"]:
                    st.markdown(
                        f"- After round 5: **{stats['rounds'][5]['cumulative_count']}/{total_tasks}** tasks passed ({stats['rounds'][5]['cumulative_count']/total_tasks:.0%})"
                    )
                elif stats["max_round"] >= 5:
                    max_round_below_5 = max([r for r in stats["rounds"].keys() if r <= 5])
                    st.markdown(
                        f"- After round 5: **{stats['rounds'][max_round_below_5]['cumulative_count']}/{total_tasks}** tasks passed ({stats['rounds'][max_round_below_5]['cumulative_count']/total_tasks:.0%})"
                    )

                if 10 in stats["rounds"]:
                    st.markdown(
                        f"- After round 10: **{stats['rounds'][10]['cumulative_count']}/{total_tasks}** tasks passed ({stats['rounds'][10]['cumulative_count']/total_tasks:.0%})"
                    )
                elif stats["max_round"] >= 1:
                    st.markdown(
                        f"- After final round ({stats['max_round']}): **{stats['rounds'][stats['max_round']]['cumulative_count']}/{total_tasks}** tasks passed ({stats['rounds'][stats['max_round']]['cumulative_count']/total_tasks:.0%})"
                    )
    else:
        st.info("No task completion data available.")


if state.scenario is not None:
    summary_window()
    if st.toggle("show analyse_task_competition"):
        analyze_task_completion()

    # R&D Loops Window
    if isinstance(state.scenario, SIMILAR_SCENARIOS):
        st.header("R&D Loops♾️", divider="rainbow", anchor="_rdloops")
        if len(state.msgs) > 1:
            r_options = list(state.msgs.keys())
            if 0 in r_options:
                r_options.remove(0)
            round = st.radio("**Loops**", horizontal=True, options=r_options, index=state.lround - 1)
        else:
            round = 1

        rf_c, d_c = st.columns([2, 2])
    elif isinstance(state.scenario, GeneralModelScenario):

        rf_c = st.container()
        d_c = st.container()
        round = 0
    else:
        st.error("Unknown Scenario!")
        st.stop()

    with rf_c:
        research_window()
        feedback_window()

    with d_c.container(border=True):
        evolving_window()


st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("#### Disclaimer")
st.markdown(
    "*This content is AI-generated and may not be fully accurate or up-to-date; please verify with a professional for critical matters.*",
    unsafe_allow_html=True,
)
