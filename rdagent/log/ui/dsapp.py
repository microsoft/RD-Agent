import hashlib
import math
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit import session_state as state

from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.log.mle_summary import extract_mle_json, is_valid_session
from rdagent.log.storage import FileStorage
from rdagent.log.ui.conf import UI_SETTING
from rdagent.utils import remove_ansi_codes

st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")

# 设置主日志路径
if "log_folder" not in state:
    state.log_folder = Path("./log")
if "log_folders" not in state:
    state.log_folders = UI_SETTING.default_log_folders
if "log_path" not in state:
    state.log_path = None
if "show_all_summary" not in state:
    state.show_all_summary = True
if "show_stdout" not in state:
    state.show_stdout = False


def load_stdout():
    # FIXME: TODO: 使用配置项来指定stdout文件名
    stdout_path = state.log_folder / f"{state.log_path}.stdout"
    if stdout_path.exists():
        stdout = stdout_path.read_text()
    else:
        stdout = f"Please Set: {stdout_path}"
    return stdout


def extract_loopid_func_name(tag):
    """提取 Loop ID 和函数名称"""
    match = re.search(r"Loop_(\d+)\.([^.]+)", tag)
    return match.groups() if match else (None, None)


def extract_evoid(tag):
    """提取 EVO ID"""
    match = re.search(r"\.evo_loop_(\d+)\.", tag)
    return match.group(1) if match else None


# @st.cache_data
def load_data(log_path: Path):
    state.data = defaultdict(lambda: defaultdict(dict))
    state.times = defaultdict(lambda: defaultdict(dict))
    for msg in FileStorage(log_path).iter_msg():
        if msg.tag and "llm" not in msg.tag and "session" not in msg.tag:
            if msg.tag == "competition":
                state.data["competition"] = msg.content
                continue

            li, fn = extract_loopid_func_name(msg.tag)
            li = int(li)

            # read times
            loop_obj_path = log_path / "__session__" / f"{li}" / "4_record"
            if loop_obj_path.exists():
                try:
                    state.times[li] = DataScienceRDLoop.load(loop_obj_path, do_truncate=False).loop_trace[li]
                except Exception as e:
                    pass

            ei = extract_evoid(msg.tag)
            msg.tag = re.sub(r"\.evo_loop_\d+", "", msg.tag)
            msg.tag = re.sub(r"Loop_\d+\.[^.]+\.?", "", msg.tag)
            msg.tag = msg.tag.strip()

            if ei:
                if int(ei) not in state.data[li][fn]:
                    state.data[li][fn][int(ei)] = {}
                state.data[li][fn][int(ei)][msg.tag] = msg.content
            else:
                if msg.tag:
                    state.data[li][fn][msg.tag] = msg.content
                else:
                    if not isinstance(msg.content, str):
                        state.data[li][fn]["no_tag"] = msg.content


# @st.cache_data
def get_folders_sorted(log_path):
    """缓存并返回排序后的文件夹列表，并加入进度打印"""
    if not log_path.exists():
        st.toast(f"Path {log_path} does not exist!")
        return []
    with st.spinner("正在加载文件夹列表..."):
        folders = sorted(
            (folder for folder in log_path.iterdir() if is_valid_session(folder)),
            key=lambda folder: folder.stat().st_mtime,
            reverse=True,
        )
        st.write(f"找到 {len(folders)} 个文件夹")
    return [folder.name for folder in folders]


# UI - Sidebar
with st.sidebar:
    log_folder_str = st.text_area(
        "**Log Folders**(split by ';')", placeholder=state.log_folder, value=";".join(state.log_folders)
    )
    state.log_folders = [folder.strip() for folder in log_folder_str.split(";") if folder.strip()]

    # # TODO: 只是临时的功能
    day_map = {"srv": "最近(srv)", "srv2": "上一批(srv2)", "srv3": "上上批(srv3)"}
    day_srv = st.radio("选择批次", ["srv", "srv2", "srv3"], format_func=lambda x: day_map[x], horizontal=True)
    if day_srv == "srv":
        state.log_folders = [re.sub(r"log\.srv\d*", "log.srv", folder) for folder in state.log_folders]
    elif day_srv == "srv2":
        state.log_folders = [re.sub(r"log\.srv\d*", "log.srv2", folder) for folder in state.log_folders]
    elif day_srv == "srv3":
        state.log_folders = [re.sub(r"log\.srv\d*", "log.srv3", folder) for folder in state.log_folders]

    state.log_folder = Path(st.radio(f"Select :blue[**one log folder**]", state.log_folders))
    if not state.log_folder.exists():
        st.warning(f"Path {state.log_folder} does not exist!")
    else:
        folders = get_folders_sorted(state.log_folder)
        st.selectbox(f"Select from :blue[**{state.log_folder.absolute()}**]", folders, key="log_path")

        if st.button("Refresh Data"):
            if state.log_path is None:
                st.toast("Please select a log path first!", icon="🟡")
                st.stop()

            load_data(state.log_folder / state.log_path)

    st.toggle("One Trace / Log Folder Summary", key="show_all_summary")
    st.toggle("Show stdout", key="show_stdout")


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


def workspace_win(data, instance_id=None):
    show_files = {k: v for k, v in data.file_dict.items() if "test" not in k}

    base_key = str(data.workspace_path)
    if instance_id is not None:
        base_key += f"_{instance_id}"
    unique_key = hashlib.md5(base_key.encode()).hexdigest()

    if len(show_files) > 0:
        with st.expander(f"Files in :blue[{replace_ep_path(data.workspace_path)}]"):
            code_tabs = st.tabs(show_files.keys())
            for ct, codename in zip(code_tabs, show_files.keys()):
                with ct:
                    st.code(
                        show_files[codename],
                        language=("python" if codename.endswith(".py") else "markdown"),
                        wrap_lines=True,
                        line_numbers=True,
                    )

            st.markdown("### Save All Files to Folder")
            target_folder = st.text_input("Enter target folder path:", key=f"save_folder_path_input_{unique_key}")

            if st.button("Save Files", key=f"save_files_button_{unique_key}"):
                if target_folder.strip() == "":
                    st.warning("Please enter a valid folder path.")
                else:
                    target_folder_path = Path(target_folder)
                    target_folder_path.mkdir(parents=True, exist_ok=True)
                    for filename, content in data.file_dict.items():
                        save_path = target_folder_path / filename
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        save_path.write_text(content, encoding="utf-8")
                    st.success(f"All files saved to: {target_folder}")
    else:
        st.markdown(f"No files in :blue[{replace_ep_path(data.workspace_path)}]")


def hypothesis_win(data):
    st.code(str(data).replace("\n", "\n\n"), wrap_lines=True)


def exp_gen_win(data):
    st.header("Exp Gen", divider="blue", anchor="exp-gen")
    st.subheader("Hypothesis")
    hypothesis_win(data["no_tag"].hypothesis)

    st.subheader("pending_tasks")
    for tasks in data["no_tag"].pending_tasks_list:
        task_win(tasks[0])
    st.subheader("Exp Workspace")
    workspace_win(data["no_tag"].experiment_workspace, instance_id="exp_gen")


def evolving_win(data, key):
    with st.container(border=True):
        if len(data) > 1:
            evo_id = st.slider("Evolving", 0, len(data) - 1, 0, key=key)
        elif len(data) == 1:
            evo_id = 0
        else:
            st.markdown("No evolving.")
            return

        if evo_id in data:
            if data[evo_id]["evolving code"][0] is not None:
                st.subheader("codes")
                workspace_win(data[evo_id]["evolving code"][0], instance_id=key)
                fb = data[evo_id]["evolving feedback"][0]
                st.subheader("evolving feedback" + ("✅" if bool(fb) else "❌"))
                f1, f2, f3 = st.tabs(["execution", "return_checking", "code"])
                f1.code(fb.execution, wrap_lines=True)
                f2.code(fb.return_checking, wrap_lines=True)
                f3.code(fb.code, wrap_lines=True)
            else:
                st.write("data[evo_id]['evolving code'][0] is None.")
                st.write(data[evo_id])
        else:
            st.markdown("No evolving.")


def coding_win(data):
    st.header("Coding", divider="blue", anchor="coding")
    evolving_data = {k: v for k, v in data.items() if isinstance(k, int)}
    task_set = set()
    for v in evolving_data.values():
        for t in v:
            if "Task" in t.split(".")[0]:
                task_set.add(t.split(".")[0])
    if task_set:
        # 新版存Task tag的Trace
        for task in task_set:
            st.subheader(task)
            task_data = {k: {a.split(".")[1]: b for a, b in v.items() if task in a} for k, v in evolving_data.items()}
            evolving_win(task_data, key=task)
    else:
        # 旧版未存Task tag的Trace
        evolving_win(evolving_data, key="coding")
    if "no_tag" in data:
        st.subheader("Exp Workspace (coding final)")
        workspace_win(data["no_tag"].experiment_workspace, instance_id="coding")


def running_win(data, mle_score):
    st.header("Running", divider="blue", anchor="running")
    evolving_win({k: v for k, v in data.items() if isinstance(k, int)}, key="running")
    if "no_tag" in data:
        st.subheader("Exp Workspace (running final)")
        workspace_win(data["no_tag"].experiment_workspace, instance_id="running_dump")
        st.subheader("Result")
        st.write(data["no_tag"].result)
        st.subheader("MLE Submission Score" + ("✅" if (isinstance(mle_score, dict) and mle_score["score"]) else "❌"))
        if isinstance(mle_score, dict):
            st.json(mle_score)
        else:
            st.code(mle_score, wrap_lines=True)


def feedback_win(data):
    data = data["no_tag"]
    st.header("Feedback" + ("✅" if bool(data) else "❌"), divider="orange", anchor="feedback")
    st.code(str(data).replace("\n", "\n\n"), wrap_lines=True)
    if data.exception is not None:
        st.markdown(f"**:red[Exception]**: {data.exception}")


def sota_win(data):
    st.header("SOTA Experiment", divider="rainbow", anchor="sota-exp")
    if data:
        st.markdown(f"**SOTA Exp Hypothesis**")
        hypothesis_win(data.hypothesis)
        st.markdown("**Exp Workspace**")
        workspace_win(data.experiment_workspace, instance_id="sota")
    else:
        st.markdown("No SOTA experiment.")


def main_win(data):
    exp_gen_win(data["direct_exp_gen"])
    if "coding" in data:
        coding_win(data["coding"])
    if "running" in data:
        running_win(data["running"], data["mle_score"])
    if "feedback" in data:
        feedback_win(data["feedback"])
    if "record" in data and "SOTA experiment" in data["record"]:
        sota_win(data["record"]["SOTA experiment"])

    with st.sidebar:
        st.markdown(
            f"""
- [Exp Gen](#exp-gen)
- [Coding](#coding)
- [Running](#running)
- [Feedback](#feedback)
- [SOTA Experiment](#sota-exp)
"""
        )


def replace_ep_path(p: Path):
    # 替换workspace path为对应ep机器mount在ep03的path
    # TODO: FIXME: 使用配置项来处理
    match = re.search(r"ep\d+", str(state.log_folder))
    if match:
        ep = match.group(0)
        return Path(
            str(p).replace("repos/RD-Agent-Exp", f"repos/batch_ctrl/all_projects/{ep}").replace("/Data", "/data")
        )
    return p


def summarize_data():
    st.header("Summary", divider="rainbow")
    df = pd.DataFrame(
        columns=["Component", "Running Score", "Feedback", "e-loops", "Time", "Start Time (UTC+8)", "End Time (UTC+8)"],
        index=range(len(state.data) - 1),
    )

    for loop in range(len(state.data) - 1):
        loop_data = state.data[loop]
        df.loc[loop, "Component"] = loop_data["direct_exp_gen"]["no_tag"].hypothesis.component
        if state.times[loop]:
            df.loc[loop, "Time"] = str(sum((i.end - i.start for i in state.times[loop]), timedelta())).split(".")[0]
            df.loc[loop, "Start Time (UTC+8)"] = state.times[loop][0].start + timedelta(hours=8)
            df.loc[loop, "End Time (UTC+8)"] = state.times[loop][-1].end + timedelta(hours=8)
        if "running" in loop_data and "no_tag" in loop_data["running"]:
            if "mle_score" not in state.data[loop]:
                if "mle_score" in loop_data["running"]:
                    mle_score_txt = loop_data["running"]["mle_score"]
                    state.data[loop]["mle_score"] = extract_mle_json(mle_score_txt)
                    if state.data[loop]["mle_score"]["score"] is not None:
                        df.loc[loop, "Running Score"] = str(state.data[loop]["mle_score"]["score"])
                    else:
                        state.data[loop]["mle_score"] = mle_score_txt
                        df.loc[loop, "Running Score"] = "❌"
                else:
                    mle_score_path = (
                        replace_ep_path(loop_data["running"]["no_tag"].experiment_workspace.workspace_path)
                        / "mle_score.txt"
                    )
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
                if isinstance(state.data[loop]["mle_score"], dict):
                    df.loc[loop, "Running Score"] = str(state.data[loop]["mle_score"]["score"])
                else:
                    df.loc[loop, "Running Score"] = "❌"

        else:
            df.loc[loop, "Running Score"] = "N/A"

        if "coding" in loop_data:
            df.loc[loop, "e-loops"] = max(i for i in loop_data["coding"].keys() if isinstance(i, int)) + 1
        if "feedback" in loop_data:
            df.loc[loop, "Feedback"] = "✅" if bool(loop_data["feedback"]["no_tag"]) else "❌"
        else:
            df.loc[loop, "Feedback"] = "N/A"
    stat_t0, stat_t1 = st.columns(2)
    stat_t0.dataframe(df)

    def comp_stat_func(x: pd.DataFrame):
        total_num = x.shape[0]
        valid_num = x[x["Running Score"] != "N/A"].shape[0]
        avg_e_loops = x["e-loops"].mean()
        return pd.Series(
            {
                "Total": total_num,
                "Valid": valid_num,
                "Valid Rate": round(valid_num / total_num * 100, 2),
                "Avg e-loops": round(avg_e_loops, 2),
            }
        )

    comp_df = df.loc[:, ["Component", "Running Score", "e-loops"]].groupby("Component").apply(comp_stat_func)
    comp_df.loc["Total"] = comp_df.sum()
    comp_df.loc["Total", "Valid Rate"] = round(comp_df.loc["Total", "Valid"] / comp_df.loc["Total", "Total"] * 100, 2)
    comp_df["Valid Rate"] = comp_df["Valid Rate"].apply(lambda x: f"{x}%")
    comp_df.loc["Total", "Avg e-loops"] = round(df["e-loops"].mean(), 2)
    stat_t1.dataframe(comp_df)


def get_summary_df(log_folders: list[str]) -> tuple[dict, pd.DataFrame]:
    summarys = {}
    for lf in log_folders:
        if not (Path(lf) / "summary.pkl").exists():
            st.warning(
                f"No summary file found in **{lf}**\n\nRun:`dotenv run -- python rdagent/log/mle_summary.py grade_summary --log_folder={lf}`"
            )
        else:
            summarys[lf] = pd.read_pickle(Path(lf) / "summary.pkl")

    if len(summarys) == 0:
        return {}, pd.DataFrame()

    summary = {}
    for lf, s in summarys.items():
        for k, v in s.items():
            stdout_p = Path(lf) / f"{k}.stdout"
            v["stdout"] = []
            if stdout_p.exists():
                # stdout = stdout_p.read_text()
                stdout = ""
                if "Retrying" in stdout:
                    v["stdout"].append("LLM Retry")
                if "Traceback (most recent call last):" in stdout[-10000:]:
                    v["stdout"].append("Code Error")
            v["stdout"] = ", ".join([i for i in v["stdout"] if i])

            # 调整实验名字
            if "amlt" in lf:
                summary[f"{lf[lf.rfind('amlt')+5:].split('/')[0]} - {k}"] = v
            elif "ep" in lf:
                summary[f"{lf[lf.rfind('ep'):]} - {k}"] = v
            else:
                summary[f"{lf} - {k}"] = v

    summary = {k: v for k, v in summary.items() if "competition" in v}
    base_df = pd.DataFrame(
        columns=[
            "Competition",
            "Total Loops",
            "Successful Final Decision",
            "Made Submission",
            "Valid Submission",
            "V/M",
            "Above Median",
            "Bronze",
            "Silver",
            "Gold",
            "Any Medal",
            "Best Medal",
            "SOTA Exp",
            "Ours - Base",
            "Ours vs Base",
            "SOTA Exp Score",
            "Baseline Score",
            "Bronze Threshold",
            "Silver Threshold",
            "Gold Threshold",
            "Medium Threshold",
            "stdout",
        ],
        index=summary.keys(),
    )

    # Read baseline results
    baseline_result_path = UI_SETTING.baseline_result_path
    if Path(baseline_result_path).exists():
        baseline_df = pd.read_csv(baseline_result_path)

    for k, v in summary.items():
        loop_num = v["loop_num"]
        base_df.loc[k, "Competition"] = v["competition"]
        base_df.loc[k, "Total Loops"] = loop_num
        if loop_num == 0:
            base_df.loc[k] = "N/A"
        else:
            base_df.loc[k, "Successful Final Decision"] = v["success_loop_num"]
            base_df.loc[k, "Made Submission"] = v["made_submission_num"]
            base_df.loc[k, "Valid Submission"] = v["valid_submission_num"]
            base_df.loc[k, "Above Median"] = v["above_median_num"]
            base_df.loc[k, "Bronze"] = v["bronze_num"]
            if v["bronze_num"] > 0:
                base_df.loc[k, "Best Medal"] = "bronze"
            base_df.loc[k, "Silver"] = v["silver_num"]
            if v["silver_num"] > 0:
                base_df.loc[k, "Best Medal"] = "silver"
            base_df.loc[k, "Gold"] = v["gold_num"]
            if v["gold_num"] > 0:
                base_df.loc[k, "Best Medal"] = "gold"
            base_df.loc[k, "Any Medal"] = v["get_medal_num"]

            baseline_score = None
            if Path(baseline_result_path).exists():
                baseline_score = baseline_df.loc[baseline_df["competition_id"] == v["competition"], "score"].item()

            base_df.loc[k, "SOTA Exp"] = v.get("sota_exp_stat", None)
            if baseline_score is not None and v.get("sota_exp_score", None) is not None:
                base_df.loc[k, "Ours - Base"] = v["sota_exp_score"] - baseline_score
                try:
                    base_df.loc[k, "Ours vs Base"] = math.exp(
                        abs(math.log(v["sota_exp_score"] / baseline_score))
                    )  # exp^|ln(a/b)|
                except Exception as e:
                    base_df.loc[k, "Ours vs Base"] = None
            base_df.loc[k, "SOTA Exp Score"] = v.get("sota_exp_score", None)
            base_df.loc[k, "Baseline Score"] = baseline_score
            base_df.loc[k, "Bronze Threshold"] = v.get("bronze_threshold", None)
            base_df.loc[k, "Silver Threshold"] = v.get("silver_threshold", None)
            base_df.loc[k, "Gold Threshold"] = v.get("gold_threshold", None)
            base_df.loc[k, "Medium Threshold"] = v.get("median_threshold", None)
            base_df.loc[k, "stdout"] = v["stdout"]

    base_df["SOTA Exp"].replace("", pd.NA, inplace=True)
    base_df = base_df.astype(
        {
            "Total Loops": int,
            "Successful Final Decision": int,
            "Made Submission": int,
            "Valid Submission": int,
            "Above Median": int,
            "Bronze": int,
            "Silver": int,
            "Gold": int,
            "Any Medal": int,
            "Ours - Base": float,
            "Ours vs Base": float,
            "SOTA Exp Score": float,
            "Baseline Score": float,
            "Bronze Threshold": float,
            "Silver Threshold": float,
            "Gold Threshold": float,
            "Medium Threshold": float,
        }
    )
    return summary, base_df


def num2percent(num: int, total: int, show_origin=True) -> str:
    if show_origin:
        return f"{num} ({round(num / total * 100, 2)}%)"
    return f"{round(num / total * 100, 2)}%"


def percent_df(df: pd.DataFrame, show_origin=True) -> pd.DataFrame:
    base_df = df.astype("object", copy=True)
    for k in base_df.index:
        loop_num = int(base_df.loc[k, "Total Loops"])
        if loop_num != 0:
            base_df.loc[k, "Successful Final Decision"] = num2percent(
                base_df.loc[k, "Successful Final Decision"], loop_num, show_origin
            )
            if base_df.loc[k, "Made Submission"] != 0:
                base_df.loc[k, "V/M"] = (
                    f"{round(base_df.loc[k, 'Valid Submission'] / base_df.loc[k, 'Made Submission'] * 100, 2)}%"
                )
            else:
                base_df.loc[k, "V/M"] = "N/A"
            base_df.loc[k, "Made Submission"] = num2percent(base_df.loc[k, "Made Submission"], loop_num, show_origin)
            base_df.loc[k, "Valid Submission"] = num2percent(base_df.loc[k, "Valid Submission"], loop_num, show_origin)
            base_df.loc[k, "Above Median"] = num2percent(base_df.loc[k, "Above Median"], loop_num, show_origin)
            base_df.loc[k, "Bronze"] = num2percent(base_df.loc[k, "Bronze"], loop_num, show_origin)
            base_df.loc[k, "Silver"] = num2percent(base_df.loc[k, "Silver"], loop_num, show_origin)
            base_df.loc[k, "Gold"] = num2percent(base_df.loc[k, "Gold"], loop_num, show_origin)
            base_df.loc[k, "Any Medal"] = num2percent(base_df.loc[k, "Any Medal"], loop_num, show_origin)
    return base_df


def days_summarize_win():
    lfs1 = [re.sub(r"log\.srv\d*", "log.srv", folder) for folder in state.log_folders]
    lfs2 = [re.sub(r"log\.srv\d*", "log.srv2", folder) for folder in state.log_folders]
    lfs3 = [re.sub(r"log\.srv\d*", "log.srv3", folder) for folder in state.log_folders]

    _, df1 = get_summary_df(lfs1)
    _, df2 = get_summary_df(lfs2)
    _, df3 = get_summary_df(lfs3)

    df = pd.concat([df1, df2, df3], axis=0)

    def mean_func(x: pd.DataFrame):
        numeric_cols = x.select_dtypes(include=["int", "float"]).mean()
        string_cols = x.select_dtypes(include=["object"]).agg(lambda col: ", ".join(col.fillna("none").astype(str)))
        return pd.concat([numeric_cols, string_cols], axis=0).reindex(x.columns).drop("Competition")

    df = df.groupby("Competition").apply(mean_func)
    if st.toggle("Show Percent", key="show_percent"):
        st.dataframe(percent_df(df, show_origin=False))
    else:
        st.dataframe(df)


def all_summarize_win():
    def shorten_folder_name(folder: str) -> str:
        if "amlt" in folder:
            return folder[folder.rfind("amlt") + 5 :].split("/")[0]
        if "ep" in folder:
            return folder[folder.rfind("ep") :]
        return folder

    selected_folders = st.multiselect(
        "Show these folders", state.log_folders, state.log_folders, format_func=shorten_folder_name
    )
    summary, base_df = get_summary_df(selected_folders)
    if not summary:
        return

    base_df = percent_df(base_df)
    st.dataframe(base_df)
    st.markdown("Ours vs Base: `math.exp(abs(math.log(sota_exp_score / baseline_score)))`")
    st.markdown(f"**统计的比赛数目: :red[{base_df.shape[0]}]**")
    total_stat = (
        base_df[
            [
                "Made Submission",
                "Valid Submission",
                "Above Median",
                "Bronze",
                "Silver",
                "Gold",
                "Any Medal",
            ]
        ]
        != "0 (0.0%)"
    ).sum()
    total_stat.name = "总体统计(%)"
    total_stat.loc["Bronze"] = base_df["Best Medal"].value_counts().get("bronze", 0)
    total_stat.loc["Silver"] = base_df["Best Medal"].value_counts().get("silver", 0)
    total_stat.loc["Gold"] = base_df["Best Medal"].value_counts().get("gold", 0)
    total_stat = total_stat / base_df.shape[0] * 100

    # SOTA Exp 统计
    se_counts = base_df["SOTA Exp"].value_counts(dropna=True)
    se_counts.loc["made_submission"] = se_counts.sum()
    se_counts.loc["Any Medal"] = se_counts.get("gold", 0) + se_counts.get("silver", 0) + se_counts.get("bronze", 0)
    se_counts.loc["above_median"] = se_counts.get("above_median", 0) + se_counts.get("Any Medal", 0)
    se_counts.loc["valid_submission"] = se_counts.get("valid_submission", 0) + se_counts.get("above_median", 0)

    sota_exp_stat = pd.Series(index=total_stat.index, dtype=int, name="SOTA Exp 统计(%)")
    sota_exp_stat.loc["Made Submission"] = se_counts.get("made_submission", 0)
    sota_exp_stat.loc["Valid Submission"] = se_counts.get("valid_submission", 0)
    sota_exp_stat.loc["Above Median"] = se_counts.get("above_median", 0)
    sota_exp_stat.loc["Bronze"] = se_counts.get("bronze", 0)
    sota_exp_stat.loc["Silver"] = se_counts.get("silver", 0)
    sota_exp_stat.loc["Gold"] = se_counts.get("gold", 0)
    sota_exp_stat.loc["Any Medal"] = se_counts.get("Any Medal", 0)
    sota_exp_stat = sota_exp_stat / base_df.shape[0] * 100

    stat_df = pd.concat([total_stat, sota_exp_stat], axis=1)
    stat_t0, stat_t1 = st.columns(2)
    with stat_t0:
        st.dataframe(stat_df.round(2))
        markdown_table = f"""
| xxx | {stat_df.iloc[0,1]:.1f} | {stat_df.iloc[1,1]:.1f} | {stat_df.iloc[2,1]:.1f} | {stat_df.iloc[3,1]:.1f} | {stat_df.iloc[4,1]:.1f} | {stat_df.iloc[5,1]:.1f} | {stat_df.iloc[6,1]:.1f}   |
"""
        st.text(markdown_table)
    with stat_t1:
        Loop_counts = base_df["Total Loops"]
        fig = px.histogram(Loop_counts, nbins=10, title="Total Loops Histogram (nbins=10)")
        mean_value = Loop_counts.mean()
        median_value = Loop_counts.median()
        fig.add_vline(
            x=mean_value, line_color="orange", annotation_text="Mean", annotation_position="top right", line_width=3
        )
        fig.add_vline(
            x=median_value, line_color="red", annotation_text="Median", annotation_position="top right", line_width=3
        )
        st.plotly_chart(fig)

    # write curve
    for k, v in summary.items():
        with st.container(border=True):
            st.markdown(f"**:blue[{k}] - :violet[{v['competition']}]**")
            fc1, fc2 = st.columns(2)
            tscores = {f"loop {k-1}": v for k, v in v["test_scores"].items()}
            tdf = pd.Series(tscores, name="score")
            f2 = px.line(tdf, markers=True, title="Test scores")
            fc2.plotly_chart(f2, key=k)
            try:
                vscores = {k: v.iloc[:, 0] for k, v in v["valid_scores"].items()}

                if len(vscores) > 0:
                    metric_name = list(vscores.values())[0].name
                else:
                    metric_name = "None"

                vdf = pd.DataFrame(vscores)
                vdf.columns = [f"loop {i}" for i in vdf.columns]
                f1 = px.line(vdf.T, markers=True, title=f"Valid scores (metric: {metric_name})")

                fc1.plotly_chart(f1, key=f"{k}_v")
            except Exception as e:
                import traceback

                st.markdown("- Error: " + str(e))
                st.code(traceback.format_exc())
                st.markdown("- Valid Scores: ")
                # st.write({k: type(v) for k, v in v["valid_scores"].items()})
                st.json(v["valid_scores"])


def stdout_win(loop_id: int):
    stdout = load_stdout()
    if stdout.startswith("Please Set"):
        st.toast(stdout, icon="🟡")
        return
    start_index = stdout.find(f"Start Loop {loop_id}")
    end_index = stdout.find(f"Start Loop {loop_id + 1}")
    loop_stdout = remove_ansi_codes(stdout[start_index:end_index])
    with st.container(border=True):
        st.subheader(f"Loop {loop_id} stdout")
        pattern = f"Start Loop {loop_id}, " + r"Step \d+: \w+"
        matches = re.finditer(pattern, loop_stdout)
        step_stdouts = {}
        for match in matches:
            step = match.group(0)
            si = match.start()
            ei = loop_stdout.find(f"Start Loop {loop_id}", match.end())
            step_stdouts[step] = loop_stdout[si:ei].strip()

        for k, v in step_stdouts.items():
            with st.expander(k, expanded=False):
                st.code(v, language="log", wrap_lines=True)


# UI - Main
if state.show_all_summary:
    with st.container(border=True):
        if st.toggle("近3天平均", key="show_3days"):
            days_summarize_win()
    with st.container(border=True):
        all_summarize_win()
elif "data" in state:
    st.title(state.data["competition"])
    summarize_data()
    if len(state.data) > 2:
        loop_id = st.slider("Loop", 0, len(state.data) - 2, 0)
    else:
        loop_id = 0
    if state.show_stdout:
        stdout_win(loop_id)
    main_win(state.data[loop_id])
