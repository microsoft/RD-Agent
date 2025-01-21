import streamlit as st
from pathlib import Path
from rdagent.log.storage import FileStorage
from streamlit import session_state as state
from collections import defaultdict
from rdagent.utils.env import MLEBDockerConf, DockerEnv
import pandas as pd
from rdagent.app.data_science.conf import DS_RD_SETTING

st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")

# 设置主日志路径
if "log_folder" not in state:
    state.log_folder = Path("./log")
if "log_path" not in state:
    state.log_path = None

# @st.cache_data
def load_data(log_path):
    data = defaultdict(lambda: defaultdict(dict))
    li = -1 # loop id
    ei = -1 # evo id
    for msg in FileStorage(state.log_folder / log_path).iter_msg():
        if msg.tag and "llm" not in msg.tag and "session" not in msg.tag:
            if msg.tag == "competition":
                data["competition"] = msg.content
                continue
            if msg.tag == "direct_exp_gen":
                li += 1
                ei = -1
            
            if "evolving " in msg.tag:
                if "evolving code" in msg.tag:
                    ei += 1
                data[li][ei][msg.tag] = msg.content
            else:
                data[li][msg.tag] = msg.content
    
    return data

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

        state.data = load_data(state.log_path)
        st.rerun()


# UI windows
def task_win(data):
    with st.container(border=True):
        st.markdown(f"**:violet[{data.name}]**")
        st.markdown(data.description)
        if hasattr(data, "architecture"): # model task
            st.markdown(f"""
    | Model_type | Architecture | hyperparameters |
    |------------|--------------|-----------------|
    | {data.model_type} | {data.architecture} | {data.hyperparameters} |
    """)

def workspace_win(data):
    show_files = {k: v for k, v in data.file_dict.items() if not "test" in k}
    if len(show_files) > 0:
        with st.expander(f"Files in :blue[{data.workspace_path}]"):
            code_tabs = st.tabs(show_files.keys())
            for ct, codename in zip(code_tabs, show_files.keys()):
                with ct:
                    st.code(show_files[codename], language=("python" if codename.endswith(".py") else "markdown"), wrap_lines=True)
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
        evo_id = st.slider("Evolving", 0, len(data)-1, 0)
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
    st.subheader("MLE Submission Score")
    st.write(mle_score)

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
        st.markdown(f"""
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
""")


def summarize_data():
    st.header("Summary", divider="rainbow")
    df = pd.DataFrame(columns=["Component", "Running", "Feedback"], index=range(len(state.data)-1))
    for loop in range(len(state.data)-1):
        loop_data = state.data[loop]
        df.loc[loop, "Component"] = loop_data["direct_exp_gen"].hypothesis.component
        if "running" in loop_data:
            if "mle_score" not in state.data[loop]:
                mle_de_conf = MLEBDockerConf()
                mle_de_conf.extra_volumes = {
                    f"{DS_RD_SETTING.local_data_path}/zip_files": "/mle/data",
                }
                de = DockerEnv(conf=mle_de_conf)
                de.prepare()
                try:
                    grade_output = loop_data["running"].experiment_workspace.execute(env=de, entry=f"mlebench grade-sample submission.csv {state.data['competition']} --data-dir /mle/data")
                    state.data[loop]["mle_score"] = grade_output
                except PermissionError:
                    state.data[loop]["mle_score"] = "No permission to access the workspace path."
                except Exception as e:
                    state.data[loop]["mle_score"] = e

            df.loc[loop, "Running"] = "✅"
        else:
            df.loc[loop, "Running"] = "N/A"
        if "feedback" in loop_data:
            df.loc[loop, "Feedback"] = "✅" if bool(loop_data["feedback"]) else "❌"
        else:
            df.loc[loop, "Feedback"] = "N/A"
    st.dataframe(df)

# UI - Main
if "data" in state:
    st.title(state.data["competition"])
    summarize_data()
    loop_id = st.slider("Loop", 0, len(state.data)-2, 0)
    main_win(state.data[loop_id])
