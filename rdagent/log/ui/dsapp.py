import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit import session_state as state

from rdagent.log.mle_summary import extract_mle_json
from rdagent.log.storage import FileStorage

st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")

# 设置主日志路径
if "log_folder" not in state:
    state.log_folder = Path("./log")
if "log_path" not in state:
    state.log_path = None
if "show_all_summary" not in state:
    state.show_all_summary = True


def extract_loopid_func_name(tag):
    """提取 Loop ID 和函数名称"""
    match = re.search(r"Loop_(\d+)\.([^.]+)", tag)
    return match.groups() if match else (None, None)


def extract_evoid(tag):
    """提取 EVO ID"""
    match = re.search(r"\.evo_loop_(\d+)\.", tag)
    return match.group(1) if match else None


# @st.cache_data
def load_data(log_path):
    state.data = defaultdict(lambda: defaultdict(dict))
    for msg in FileStorage(state.log_folder / log_path).iter_msg():
        if msg.tag and "llm" not in msg.tag and "session" not in msg.tag:
            if msg.tag == "competition":
                state.data["competition"] = msg.content
                continue

            li, fn = extract_loopid_func_name(msg.tag)
            li = int(li)
            ei = extract_evoid(msg.tag)
            msg.tag = re.sub(r"\.evo_loop_\d+", "", msg.tag)
            msg.tag = re.sub(r"Loop_\d+\.[^.]+\.?", "", msg.tag)
            msg.tag = msg.tag.strip()

            if ei:
                state.data[li][int(ei)][msg.tag] = msg.content
            else:
                if msg.tag:
                    state.data[li][fn][msg.tag] = msg.content
                else:
                    if not isinstance(msg.content, str):
                        state.data[li][fn] = msg.content


@st.cache_data
def get_folders_sorted(log_path):
    """缓存并返回排序后的文件夹列表，并加入进度打印"""
    with st.spinner("正在加载文件夹列表..."):
        folders = sorted(
            (folder for folder in log_path.iterdir() if folder.is_dir() and list(folder.iterdir())),
            key=lambda folder: folder.stat().st_mtime,
            reverse=True,
        )
        st.write(f"找到 {len(folders)} 个文件夹")
    return [folder.name for folder in folders]


# UI - Sidebar
with st.sidebar:
    state.log_folder = Path(st.text_input("**Log Folder**", placeholder=state.log_folder, value=state.log_folder))
    if not state.log_folder.exists():
        st.warning(f"Path {state.log_folder} does not exist!")
    folders = get_folders_sorted(state.log_folder)
    st.selectbox(f"Select from :blue[**{state.log_folder.absolute()}**]", folders, key="log_path")

    if st.button("Refresh Data"):
        if state.log_path is None:
            st.toast("Please select a log path first!", type="error")
            st.stop()

        load_data(state.log_path)

    st.toggle("One Trace / Log Folder Summary", key="show_all_summary")


# UI windows
def task_win(data):
    with st.container(border=True):
        st.markdown(f"**:violet[{data.name}]**")
        st.markdown(data.description)
        if hasattr(data, "architecture"):  # model task
            st.markdown(
                f"""
    | Model_type | Architecture | hyperparameters |
    |------------|--------------|-----------------|
    | {data.model_type} | {data.architecture} | {data.hyperparameters} |
    """
            )


def workspace_win(data):
    show_files = {k: v for k, v in data.file_dict.items() if not "test" in k}
    if len(show_files) > 0:
        with st.expander(f"Files in :blue[{data.workspace_path}]"):
            code_tabs = st.tabs(show_files.keys())
            for ct, codename in zip(code_tabs, show_files.keys()):
                with ct:
                    st.code(
                        show_files[codename],
                        language=("python" if codename.endswith(".py") else "markdown"),
                        wrap_lines=True,
                    )
    else:
        st.markdown("No files in the workspace")


def exp_gen_win(data):
    st.header("Exp Gen", divider="blue")
    st.subheader("Hypothesis")
    st.markdown(data.hypothesis)

    st.subheader("pending_tasks")
    for tasks in data.pending_tasks_list:
        task_win(tasks[0])
    st.subheader("Exp Workspace", anchor="exp-workspace")
    workspace_win(data.experiment_workspace)


def evolving_win(data):
    st.header("Code Evolving", divider="green")
    if len(data) > 1:
        evo_id = st.slider("Evolving", 0, len(data) - 1, 0)
    else:
        evo_id = 0

    if evo_id in data:
        st.subheader("codes")
        workspace_win(data[evo_id]["evolving code"][0])
        fb = data[evo_id]["evolving feedback"][0]
        st.subheader("evolving feedback" + ("✅" if bool(fb) else "❌"), anchor="c_feedback")
        f1, f2, f3 = st.tabs(["execution", "return_checking", "code"])
        f1.code(fb.execution, wrap_lines=True)
        f2.code(fb.return_checking, wrap_lines=True)
        f3.code(fb.code, wrap_lines=True)
    else:
        st.markdown("No evolving.")


def exp_after_coding_win(data):
    st.header("Exp After Coding", divider="blue")
    st.subheader("Exp Workspace", anchor="eac-exp-workspace")
    workspace_win(data.experiment_workspace)


def exp_after_running_win(data, mle_score):
    st.header("Exp After Running", divider="blue")
    st.subheader("Exp Workspace", anchor="ear-exp-workspace")
    workspace_win(data.experiment_workspace)
    st.subheader("Result")
    st.write(data.result)
    st.subheader("MLE Submission Score" + ("✅" if (isinstance(mle_score, dict) and mle_score["score"]) else "❌"))
    if isinstance(mle_score, dict):
        st.json(mle_score)
    else:
        st.code(mle_score, wrap_lines=True)


def feedback_win(data):
    st.header("Feedback" + ("✅" if bool(data) else "❌"), divider="orange")
    st.code(data, wrap_lines=True)
    if data.exception is not None:
        st.markdown(f"**:red[Exception]**: {data.exception}")


def sota_win(data):
    st.header("SOTA Experiment", divider="rainbow")
    if data:
        st.subheader("Exp Workspace", anchor="sota-exp-workspace")
        workspace_win(data.experiment_workspace)
    else:
        st.markdown("No SOTA experiment.")


def main_win(data):
    exp_gen_win(data["direct_exp_gen"])
    evo_data = {k: v for k, v in data.items() if isinstance(k, int)}
    evolving_win(evo_data)
    if "coding" in data:
        exp_after_coding_win(data["coding"])
    if "running" in data:
        exp_after_running_win(data["running"], data["mle_score"])
    if "feedback" in data:
        feedback_win(data["feedback"])
    sota_win(data["SOTA experiment"])

    with st.sidebar:
        st.markdown(
            f"""
- [Exp Gen](#exp-gen)
    - [Hypothesis](#hypothesis)
    - [pending_tasks](#pending-tasks)
    - [Exp Workspace](#exp-workspace)
- [Code Evolving ({len(evo_data)})](#code-evolving)
    - [codes](#codes)
    - [evolving feedback](#c_feedback)
{"- [Exp After Coding](#exp-after-coding)" if "coding" in data else ""}
{"- [Exp After Running](#exp-after-running)" if "running" in data else ""}
{"- [Feedback](#feedback)" if "feedback" in data else ""}
- [SOTA Experiment](#sota-experiment)
"""
        )


def summarize_data():
    st.header("Summary", divider="rainbow")
    df = pd.DataFrame(columns=["Component", "Running Score", "Feedback"], index=range(len(state.data) - 1))

    for loop in range(len(state.data) - 1):
        loop_data = state.data[loop]
        df.loc[loop, "Component"] = loop_data["direct_exp_gen"].hypothesis.component

        if "running" in loop_data:
            if "mle_score" not in state.data[loop]:
                mle_score_path = loop_data["running"].experiment_workspace.workspace_path / "mle_score.txt"
                try:
                    mle_score_txt = mle_score_path.read_text()
                    state.data[loop]["mle_score"] = extract_mle_json(mle_score_txt)
                    if state.data[loop]["mle_score"]["score"] is not None:
                        df.loc[loop, "Running Score"] = str(state.data[loop]["mle_score"]["score"])
                    else:
                        state.data[loop]["mle_score"] = mle_score_txt
                        df.loc[loop, "Running Score"] = "❌"
                except Exception as e:
                    state.data[loop]["mle_score"] = str(e)
                    df.loc[loop, "Running Score"] = "❌"
        else:
            df.loc[loop, "Running Score"] = "N/A"

        if "feedback" in loop_data:
            df.loc[loop, "Feedback"] = "✅" if bool(loop_data["feedback"]) else "❌"
        else:
            df.loc[loop, "Feedback"] = "N/A"
    st.dataframe(df)


def all_summarize_win():
    if not (state.log_folder / "summary.pkl").exists():
        st.warning(
            f"No summary file found in {state.log_folder}\nRun:`dotenv run -- python rdagent/log/mle_summary.py grade_summary --log_folder=<your trace folder>`"
        )
        return
    summary = pd.read_pickle(state.log_folder / "summary.pkl")
    summary = {k: v for k, v in summary.items() if "competition" in v}
    base_df = pd.DataFrame(
        columns=["Competition", "Total Loops", "Made Submission", "Successful Final Decision", "Medal"],
        index=summary.keys(),
    )
    for k, v in summary.items():
        loop_num = v["loop_num"]
        base_df.loc[k, "Competition"] = v["competition"]
        base_df.loc[k, "Total Loops"] = loop_num
        base_df.loc[k, "Made Submission"] = (
            f"{v['made_submission_num']} ({round(v['made_submission_num'] / loop_num * 100, 2)}%)"
        )
        base_df.loc[k, "Successful Final Decision"] = (
            f"{v['success_loop_num']} ({round(v['success_loop_num'] / loop_num * 100, 2)}%)"
        )
        base_df.loc[k, "Medal"] = v["medal"]
    st.dataframe(base_df)
    # write curve
    for k, v in summary.items():
        with st.container(border=True):
            st.markdown(f"**:blue[{k}] - :violet[{v['competition']}]**")
            vscores = {k: v.iloc[:, 0] for k, v in v["valid_scores"].items()}
            if len(vscores) > 0:
                metric_name = list(vscores.values())[0].name
            else:
                metric_name = "None"

            fc1, fc2 = st.columns(2)
            vdf = pd.DataFrame(vscores)
            vdf.columns = [f"loop {i}" for i in vdf.columns]
            f1 = px.line(vdf.T, markers=True, title=f"Valid scores (metric: {metric_name})")
            fc1.plotly_chart(f1, key=f"{k}_v")

            tscores = {f"loop {k}": v for k, v in v["test_scores"].items()}
            tdf = pd.Series(tscores, name="score")
            f2 = px.line(tdf, markers=True, title="Test scores")
            fc2.plotly_chart(f2, key=k)


# UI - Main
if state.show_all_summary:
    all_summarize_win()
elif "data" in state:
    st.title(state.data["competition"])
    summarize_data()
    loop_id = st.slider("Loop", 0, len(state.data) - 2, 0)
    main_win(state.data[loop_id])
