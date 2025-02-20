import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit import session_state as state

from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.log.mle_summary import extract_mle_json, is_valid_session
from rdagent.log.storage import FileStorage
from rdagent.utils import remove_ansi_codes

st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")

# 设置主日志路径
if "log_folder" not in state:
    state.log_folder = Path("./log")
if "log_folders" not in state:
    state.log_folders = ["./log"]
if "log_path" not in state:
    state.log_path = None
if "show_all_summary" not in state:
    state.show_all_summary = True
if "show_stdout" not in state:
    state.show_stdout = False

# 移除原先的全局 multi_logs_is_lower_better；改为每个比赛都有自己的开关


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
                state.times[li] = DataScienceRDLoop.load(loop_obj_path).loop_trace[li]

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


# @st.cache_data
def get_folders_sorted(log_path):
    """缓存并返回排序后的文件夹列表，并加入进度打印"""
    with st.spinner("正在加载文件夹列表..."):
        folders = sorted(
            (folder for folder in log_path.iterdir() if is_valid_session(folder)),
            key=lambda folder: folder.stat().st_mtime,
            reverse=True,
        )
        st.write(f"找到 {len(folders)} 个文件夹")
    return [folder.name for folder in folders]


def parse_log_for_hypothesis(log_trace_path: Path) -> dict[int, dict]:
    """
    解析指定日志目录中的 loop -> {component, hypothesis} 映射
    """
    loop2hypo = {}
    for msg in FileStorage(log_trace_path).iter_msg():
        if msg.tag and "direct_exp_gen" in msg.tag:
            # direct_exp_gen 对象里是 DSExperiment
            if hasattr(msg.content, "hypothesis"):
                h = msg.content.hypothesis
                # 这里既存 component，也存 hypothesis
                m = re.search(r"Loop_(\d+)", msg.tag)
                if m:
                    li = int(m.group(1))
                    loop2hypo[li] = {
                        "component": str(h.component),
                        "hypothesis": str(h.hypothesis),  # 只存最核心的文字即可
                    }
    return loop2hypo


# UI - Sidebar
with st.sidebar:
    log_folder_str = st.text_area(
        "**Log Folders**(split by ';')", placeholder=str(state.log_folder), value=";".join(state.log_folders)
    )
    state.log_folders = [folder.strip() for folder in log_folder_str.split(";") if folder.strip()]

    state.log_folder = Path(st.radio(f"Select :blue[**one log folder**]", state.log_folders))
    if not state.log_folder.exists():
        st.warning(f"Path {state.log_folder} does not exist!")

    folders = get_folders_sorted(state.log_folder)
    st.selectbox(f"Select from :blue[**{state.log_folder.absolute()}**]", folders, key="log_path")

    if st.button("Refresh Data"):
        if state.log_path is None:
            st.toast("Please select a log path first!", type="error")
            st.stop()

        load_data(state.log_folder / state.log_path)

    st.toggle("One Trace / Log Folder Summary", key="show_all_summary")
    st.toggle("Show stdout", key="show_stdout")


def task_win(data):
    with st.container():
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
    show_files = {k: v for k, v in data.file_dict.items() if "test" not in k}
    if len(show_files) > 0:
        with st.expander(f"Files in :blue[{replace_ep_path(data.workspace_path)}]"):
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
    st.code(str(data.hypothesis).replace("\n", "\n\n"), wrap_lines=True)

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
        if data[evo_id]["evolving code"][0] is not None:
            st.subheader("codes")
            workspace_win(data[evo_id]["evolving code"][0])
            fb = data[evo_id]["evolving feedback"][0]
            st.subheader("evolving feedback" + ("✅" if bool(fb) else "❌"), anchor="c_feedback")
            f1, f2, f3 = st.tabs(["execution", "return_checking", "code"])
            f1.code(fb.execution, wrap_lines=True)
            f2.code(fb.return_checking, wrap_lines=True)
            f3.code(fb.code, wrap_lines=True)
        else:
            st.write("data[evo_id]['evolving code'][0] is None.")
            st.write(data[evo_id])
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
    """
    单日志模式下：保留最初的 Score Delta(pp) 逻辑
    """
    st.header("Summary", divider="rainbow")
    df = pd.DataFrame(
        columns=["Component", "Running Score", "Feedback", "Time", "Start Time (UTC+8)", "End Time (UTC+8)"],
        index=range(len(state.data) - 1),
    )

    df["Score Delta(pp)"] = None
    best_score = None
    is_lower_better = None

    for loop in range(len(state.data) - 1):
        loop_data = state.data[loop]
        df.loc[loop, "Component"] = loop_data["direct_exp_gen"].hypothesis.component
        if state.times[loop]:
            df.loc[loop, "Time"] = str(sum((i.end - i.start for i in state.times[loop]), timedelta())).split(".")[0]
            df.loc[loop, "Start Time (UTC+8)"] = state.times[loop][0].start + timedelta(hours=8)
            df.loc[loop, "End Time (UTC+8)"] = state.times[loop][-1].end + timedelta(hours=8)
        if "running" in loop_data:
            if "mle_score" not in loop_data:
                mle_score_path = (
                    replace_ep_path(loop_data["running"].experiment_workspace.workspace_path) / "mle_score.txt"
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
                    if is_lower_better is None:
                        is_lower_better = state.data[loop]["mle_score"].get("is_lower_better", False)
                else:
                    df.loc[loop, "Running Score"] = "❌"
        else:
            df.loc[loop, "Running Score"] = "N/A"

        if "feedback" in loop_data:
            df.loc[loop, "Feedback"] = "✅" if bool(loop_data["feedback"]) else "❌"
        else:
            df.loc[loop, "Feedback"] = "N/A"

    st.dataframe(df)

    for i in df.index:
        val = df.loc[i, "Running Score"]
        try:
            current_score = float(val)
        except:
            current_score = None

        if current_score is not None:
            if best_score is None:
                best_score = current_score
                df.loc[i, "Score Delta(pp)"] = 0.0
            else:
                if is_lower_better:
                    diff = (best_score - current_score) * 100
                    df.loc[i, "Score Delta(pp)"] = round(diff, 4)
                    if current_score < best_score:
                        best_score = current_score
                else:
                    diff = (current_score - best_score) * 100
                    df.loc[i, "Score Delta(pp)"] = round(diff, 4)
                    if current_score > best_score:
                        best_score = current_score
        else:
            df.loc[i, "Score Delta(pp)"] = None

    st.subheader("Score Delta(pp) Added")
    st.dataframe(df)

    df_plot = df.dropna(subset=["Score Delta(pp)"]).copy()
    df_plot["loop"] = df_plot.index
    if not df_plot.empty:
        fig = px.bar(
            df_plot,
            x="loop",
            y="Score Delta(pp)",
            # 4) remove numeric text on the bar
            title="Incremental Improvement (Score Delta in pp) - Single Log",
        )
        fig.update_traces(textposition="none")  # 仅鼠标悬停显示
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No valid Score Delta data to plot.")


def all_summarize_win():
    """
    多日志模式：
    1) 依旧展示原先 valid/test 的折线图；
    2) 额外对 valid/test 都画一张增量可视化柱状图，
       并在 hover_data 中展示 "component" + "hypothesis"，去掉柱上文字。
    3) 每个比赛都增加一个独立的 "Is Lower Better?" 开关
    4) 最后对 test increments 做一个表格，按贡献从大到小排序并显示 loop_idx, component, hypothesis, increment, contribution。
    5) 【新增功能】对于每个比赛，检查 "./log_baseline" 下是否有同名比赛，
       如果有则额外展示 baseline 的 valid score 和 test score 曲线，
       并将 baseline 的 test score 与主 log 的 test score 合并在一张图中对比。
    """
    summarys = {}
    for lf in state.log_folders:
        if not (Path(lf) / "summary.pkl").exists():
            st.warning(
                f"No summary file found in {lf}\nRun:dotenv run -- python rdagent/log/mle_summary.py grade_summary --log_folder=<your trace folder>"
            )
        else:
            summarys[lf] = pd.read_pickle(Path(lf) / "summary.pkl")

    if len(summarys) == 0:
        return

    # 用于存储每个比赛k -> bool(是否越小越好)
    if "multi_logs_better_map" not in state:
        state.multi_logs_better_map = {}

    folder_path_map = {}
    folder_hypo_map = {}

    for lf in state.log_folders:
        p_lf = Path(lf)
        if p_lf.exists():
            for log_trace_path in p_lf.iterdir():
                if is_valid_session(log_trace_path):
                    key = f"{lf[lf.rfind('ep'):]}{log_trace_path.name}"
                    folder_path_map[key] = log_trace_path
                    loop2hyp = parse_log_for_hypothesis(log_trace_path)
                    folder_hypo_map[key] = loop2hyp

    summary = {}
    for lf, s in summarys.items():
        for k, v in s.items():
            summary_key = f"{lf[lf.rfind('ep'):]}{k}"
            summary[summary_key] = v

    summary = {k: v for k, v in summary.items() if "competition" in v}

    # 新增：加载 baseline summary（固定在 "./log_baseline" 下）
    baseline_summary = {}
    baseline_folder = Path("./log_baseline")
    if baseline_folder.exists() and (baseline_folder / "summary.pkl").exists():
        baseline_summary = pd.read_pickle(baseline_folder / "summary.pkl")
    else:
        st.info("No baseline summary found in ./log_baseline")

    # 构造 baseline 比赛名称映射
    baseline_by_competition = {}
    for k_baseline, v_baseline in baseline_summary.items():
        if "competition" in v_baseline:
            baseline_by_competition[v_baseline["competition"]] = v_baseline

    base_df = pd.DataFrame(
        columns=[
            "Competition",
            "Total Loops",
            "Successful Final Decision",
            "Made Submission",
            "Valid Submission",
            "Above Median",
            "Bronze",
            "Silver",
            "Gold",
            "Any Medal",
        ],
        index=summary.keys(),
    )
    for k, v in summary.items():
        loop_num = v["loop_num"]
        base_df.loc[k, "Competition"] = v["competition"]
        if loop_num == 0:
            base_df.loc[k] = "N/A"
        else:
            base_df.loc[k, "Successful Final Decision"] = (
                f"{v['success_loop_num']} ({round(v['success_loop_num'] / loop_num * 100, 2)}%)"
            )
            base_df.loc[k, "Made Submission"] = (
                f"{v['made_submission_num']} ({round(v['made_submission_num'] / loop_num * 100, 2)}%)"
            )
            base_df.loc[k, "Valid Submission"] = (
                f"{v['valid_submission_num']} ({round(v['valid_submission_num'] / loop_num * 100, 2)}%)"
            )
            base_df.loc[k, "Above Median"] = (
                f"{v['above_median_num']} ({round(v['above_median_num'] / loop_num * 100, 2)}%)"
            )
            base_df.loc[k, "Bronze"] = f"{v['bronze_num']} ({round(v['bronze_num'] / loop_num * 100, 2)}%)"
            base_df.loc[k, "Silver"] = f"{v['silver_num']} ({round(v['silver_num'] / loop_num * 100, 2)}%)"
            base_df.loc[k, "Gold"] = f"{v['gold_num']} ({round(v['gold_num'] / loop_num * 100, 2)}%)"
            base_df.loc[k, "Any Medal"] = f"{v['get_medal_num']} ({round(v['get_medal_num'] / loop_num * 100, 2)}%)"

    st.dataframe(base_df)
    total_stat = (
        (
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
        / base_df.shape[0]
        * 100
    )
    total_stat.name = "总体统计(%)"
    st.dataframe(total_stat.round(2))

    # write curve
    for k, v in summary.items():
        with st.container():
            st.markdown(f"**:blue[{k}] - :violet[{v['competition']}]**")

            # 每个比赛独立设置 "Is Lower Better?"
            old_val = state.multi_logs_better_map.get(k, False)
            new_val = st.checkbox(f"Is Lower Better? ({k})", key=f"lower_better_{k}", value=old_val)
            state.multi_logs_better_map[k] = new_val

            # 恢复对 metric name 的解析
            vscores = v["valid_scores"]  # {loop_idx: DataFrame}
            if len(vscores) > 0:
                # 取第一个 df 的第一列，看它的 name
                first_df = list(vscores.values())[0]
                metric_name = first_df.columns[0] if not first_df.empty else "None"
            else:
                metric_name = "None"

            # valid vals
            vscores_dict = {}
            for loop_idx, score_df in vscores.items():
                if not score_df.empty:
                    val = score_df.iloc[0, 0]
                else:
                    val = None
                vscores_dict[loop_idx] = val

            fc1, fc2 = st.columns(2)
            # 画 valid curve
            vdf = pd.Series(vscores_dict, name=metric_name).sort_index()
            f1 = px.line(vdf, markers=True, title=f"Valid scores (metric: {metric_name})")
            fc1.plotly_chart(f1, key=f"{k}_v")

            # test scores
            tscores_dict = v["test_scores"]
            tdf = pd.Series(tscores_dict, name="test_score").sort_index()
            f2 = px.line(tdf, markers=True, title="Test scores")
            fc2.plotly_chart(f2, key=k)

            # 获取 loop->(component,hypothesis)
            loop2hypo = folder_hypo_map.get(k, {})

            # --- Valid Increment ---
            fc3, fc4 = st.columns(2)

            increments_valid = []
            best_valid = None
            for loop_idx in sorted(vscores_dict.keys()):
                val = vscores_dict[loop_idx]
                if val is not None:
                    if best_valid is None:
                        best_valid = val
                        inc = 0
                    else:
                        if new_val:  # is_lower_better
                            inc = best_valid - val
                            if val < best_valid:
                                best_valid = val
                        else:
                            inc = val - best_valid
                            if val > best_valid:
                                best_valid = val
                    increments_valid.append(
                        {
                            "loop_idx": loop_idx,
                            "increment": inc,
                            "component": loop2hypo.get(loop_idx, {}).get("component", "N/A"),
                            "hypothesis": loop2hypo.get(loop_idx, {}).get("hypothesis", "N/A"),
                        }
                    )

            if increments_valid:
                df_v_inc = pd.DataFrame(increments_valid)
                df_v_inc["Increment(pp)"] = df_v_inc["increment"] * 100.0
                fig3 = px.bar(
                    df_v_inc,
                    x="loop_idx",
                    y="Increment(pp)",
                    hover_data=["component", "hypothesis"],
                    title=f"Valid Score Increment (log: {k})",
                )
                fig3.update_traces(textposition="none")
                fc3.plotly_chart(fig3, use_container_width=True)

            # --- Test Increment ---
            increments_test = []
            best_test = None
            for loop_idx in sorted(tscores_dict.keys()):
                val = tscores_dict[loop_idx]
                if val is not None:
                    if best_test is None:
                        best_test = val
                        inc = 0
                    else:
                        if new_val:  # is_lower_better
                            inc = best_test - val
                            if val < best_test:
                                best_test = val
                        else:
                            inc = val - best_test
                            if val > best_test:
                                best_test = val
                    increments_test.append(
                        {
                            "loop_idx": loop_idx,
                            "increment": inc,
                            "component": loop2hypo.get(loop_idx, {}).get("component", "N/A"),
                            "hypothesis": loop2hypo.get(loop_idx, {}).get("hypothesis", "N/A"),
                        }
                    )

            if increments_test:
                df_t_inc = pd.DataFrame(increments_test)
                df_t_inc["Increment(pp)"] = df_t_inc["increment"] * 100.0
                fig4 = px.bar(
                    df_t_inc,
                    x="loop_idx",
                    y="Increment(pp)",
                    hover_data=["component", "hypothesis"],
                    title=f"Test Score Increment (log: {k})",
                )
                fig4.update_traces(textposition="none")
                fc4.plotly_chart(fig4, use_container_width=True)

                # 额外加一个表格，把 test increments 按贡献从大到小排列
                if new_val:
                    df_t_inc["contribution"] = -df_t_inc["increment"]
                else:
                    df_t_inc["contribution"] = df_t_inc["increment"]

                df_t_inc_sorted = df_t_inc.sort_values("contribution", ascending=False)
                st.subheader("Test Hypothesis by contribution (descending)")
                st.table(df_t_inc_sorted[["loop_idx", "component", "hypothesis", "increment", "contribution"]])

            # 【新增功能】检查 baseline 数据，并展示 baseline 的 valid/test 曲线及对比图
            competition_name = v["competition"]
            if competition_name in baseline_by_competition:
                baseline_data = baseline_by_competition[competition_name]
                # 计算 baseline valid/test 数据
                baseline_vscores_dict = {}
                for loop_idx, score_df in baseline_data["valid_scores"].items():
                    if not score_df.empty:
                        val = score_df.iloc[0, 0]
                    else:
                        val = None
                    baseline_vscores_dict[loop_idx] = val
                baseline_tscores_dict = baseline_data["test_scores"]
                # 只有在至少存在一种 baseline 数据时才显示标题及图表
                if baseline_vscores_dict or baseline_tscores_dict:
                    st.markdown("#### Baseline Log Metrics")
                    if baseline_vscores_dict:
                        baseline_vdf = pd.Series(baseline_vscores_dict, name=metric_name).sort_index()
                        fig_baseline_valid = px.line(
                            baseline_vdf,
                            markers=True,
                            title=f"Baseline Valid scores (metric: {metric_name})",
                        )
                        st.plotly_chart(fig_baseline_valid, key=f"{k}_baseline_v")
                    if baseline_tscores_dict:
                        baseline_tdf = pd.Series(baseline_tscores_dict, name="baseline_test_score").sort_index()
                        fig_baseline_test = px.line(
                            baseline_tdf,
                            markers=True,
                            title="Baseline Test scores",
                        )
                        st.plotly_chart(fig_baseline_test, key=f"{k}_baseline_t")
                        # 合并对比图：将主 log 与 baseline 的 test score 曲线合并（保证线条连续）
                        combined_loop_idx = sorted(set(list(tscores_dict.keys()) + list(baseline_tscores_dict.keys())))
                        combined_df = pd.DataFrame({
                            "loop_idx": combined_loop_idx,
                            "main_test": [tscores_dict.get(x, np.nan) for x in combined_loop_idx],
                            "baseline_test": [baseline_tscores_dict.get(x, np.nan) for x in combined_loop_idx],
                        })
                        fig_combined = go.Figure()
                        fig_combined.add_trace(go.Scatter(
                            x=combined_df["loop_idx"],
                            y=combined_df["main_test"],
                            mode='lines+markers',
                            name="Main Test Score",
                            connectgaps=True,
                        ))
                        fig_combined.add_trace(go.Scatter(
                            x=combined_df["loop_idx"],
                            y=combined_df["baseline_test"],
                            mode='lines+markers',
                            name="Baseline Test Score",
                            connectgaps=True,
                        ))
                        fig_combined.update_layout(
                            title="Combined Test Scores (Main vs Baseline)",
                            xaxis_title="Loop Index",
                            yaxis_title="Test Score",
                            legend_title="Source"
                        )
                        st.plotly_chart(fig_combined, key=f"{k}_combined_t")


def stdout_win(loop_id: int):
    stdout = load_stdout()
    if stdout.startswith("Please Set"):
        st.toast(stdout, icon="🟡")
        return
    start_index = stdout.find(f"Start Loop {loop_id}")
    end_index = stdout.find(f"Start Loop {loop_id + 1}")
    loop_stdout = remove_ansi_codes(stdout[start_index:end_index])
    with st.container():
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
            expanded = True if "coding" in k else False
            with st.expander(k, expanded=expanded):
                st.code(v, language="log", wrap_lines=True)


# UI - Main
if state.show_all_summary:
    all_summarize_win()
elif "data" in state:
    st.title(state.data["competition"])
    summarize_data()
    loop_id = st.slider("Loop", 0, len(state.data) - 2, 0)
    if state.show_stdout:
        stdout_win(loop_id)
    main_win(state.data[loop_id])
