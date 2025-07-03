import hashlib
import json
import pickle
import random
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit import session_state as state

from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.log.storage import FileStorage
from rdagent.log.ui.utils import load_times
from rdagent.log.utils import (
    LogColors,
    extract_evoid,
    extract_json,
    extract_loopid_func_name,
    is_valid_session,
)
from rdagent.utils.repo.diff import generate_diff_from_dict

if "show_stdout" not in state:
    state.show_stdout = False
if "show_llm_log" not in state:
    state.show_llm_log = False
if "data" not in state:
    state.data = defaultdict(lambda: defaultdict(dict))
if "llm_data" not in state:
    state.llm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
if "log_path" not in state:
    state.log_path = None
if "log_folder" not in state:
    state.log_folder = Path("./log")


def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d


@st.cache_data(persist=True)
def load_data(log_path: Path):
    data = defaultdict(lambda: defaultdict(dict))
    llm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for msg in FileStorage(log_path).iter_msg():
        if not msg.tag:
            continue
        li, fn = extract_loopid_func_name(msg.tag)
        ei = extract_evoid(msg.tag)
        if li:
            li = int(li)
        if ei is not None:
            ei = int(ei)
        if "debug_" in msg.tag:
            if "debug_tpl" in msg.tag and "filter_" in msg.content["uri"]:
                continue
            if ei is not None:
                llm_data[li][fn][ei].append(
                    {
                        "tag": msg.tag,
                        "obj": msg.content,
                    }
                )
            else:
                llm_data[li][fn]["no_tag"].append(
                    {
                        "tag": msg.tag,
                        "obj": msg.content,
                    }
                )
        elif "llm" not in msg.tag and "session" not in msg.tag and "batch embedding" not in msg.tag:
            if msg.tag == "competition":
                data["competition"] = msg.content
                continue
            if "SETTINGS" in msg.tag:
                data["settings"][msg.tag] = msg.content
                continue

            msg.tag = re.sub(r"\.evo_loop_\d+", "", msg.tag)
            msg.tag = re.sub(r"Loop_\d+\.[^.]+\.?", "", msg.tag)
            msg.tag = msg.tag.strip()

            if ei is not None:
                if ei not in data[li][fn]:
                    data[li][fn][ei] = {}
                data[li][fn][ei][msg.tag] = msg.content
            else:
                if msg.tag:
                    data[li][fn][msg.tag] = msg.content
                else:
                    if not isinstance(msg.content, str):
                        data[li][fn]["no_tag"] = msg.content

    # To be compatible with old version log trace, keep this
    llm_log_p = log_path / "debug_llm.pkl"
    if llm_log_p.exists():
        try:
            rd = pickle.loads(llm_log_p.read_bytes())
        except:
            rd = []
        for d in rd:
            t = d["tag"]
            if "debug_exp_gen" in t:
                continue
            if "debug_tpl" in t and "filter_" in d["obj"]["uri"]:
                continue
            lid, fn = extract_loopid_func_name(t)
            ei = extract_evoid(t)
            if lid:
                lid = int(lid)
            if ei is not None:
                ei = int(ei)

            if ei is not None:
                llm_data[lid][fn][ei].append(d)
            else:
                llm_data[lid][fn]["no_tag"].append(d)

    return convert_defaultdict_to_dict(data), convert_defaultdict_to_dict(llm_data)


def load_stdout(stdout_path: Path):
    if stdout_path.exists():
        stdout = stdout_path.read_text()
    else:
        stdout = f"Please Set: {stdout_path}"
    return stdout


# UI windows
def task_win(task):
    with st.container(border=True):
        st.markdown(f"**:violet[{task.name}]**")
        st.markdown(task.description)
        if hasattr(task, "architecture"):  # model task
            st.markdown(
                f"""
    | Model_type | Architecture | hyperparameters |
    |------------|--------------|-----------------|
    | {task.model_type} | {task.architecture} | {task.hyperparameters} |
    """
            )


def workspace_win(workspace, cmp_workspace=None, cmp_name="last code."):
    show_files = {k: v for k, v in workspace.file_dict.items() if "test" not in k}

    if len(show_files) > 0:
        if cmp_workspace:
            diff = generate_diff_from_dict(cmp_workspace.file_dict, show_files, "main.py")
            with st.expander(f":violet[**Diff with {cmp_name}**]"):
                st.code("".join(diff), language="diff", wrap_lines=True, line_numbers=True)
        with st.expander(f"Files in :blue[{replace_ep_path(workspace.workspace_path)}]"):
            code_tabs = st.tabs(show_files.keys())
            for ct, codename in zip(code_tabs, show_files.keys()):
                with ct:
                    st.code(
                        show_files[codename],
                        language=("python" if codename.endswith(".py") else "markdown"),
                        wrap_lines=True,
                        line_numbers=True,
                    )

            if state.show_save_input:
                st.markdown("### Save All Files to Folder")
                unique_key = hashlib.md5("".join(show_files.values()).encode()).hexdigest() + str(
                    random.randint(0, 10000)
                )
                target_folder = st.text_input("Enter target folder path:", key=unique_key)

                if st.button("Save Files", key=f"save_files_button_{unique_key}"):
                    if target_folder.strip() == "":
                        st.warning("Please enter a valid folder path.")
                    else:
                        target_folder_path = Path(target_folder)
                        target_folder_path.mkdir(parents=True, exist_ok=True)
                        for filename, content in workspace.file_dict.items():
                            save_path = target_folder_path / filename
                            save_path.parent.mkdir(parents=True, exist_ok=True)
                            save_path.write_text(content, encoding="utf-8")
                        st.success(f"All files saved to: {target_folder}")
    else:
        st.markdown(f"No files in :blue[{replace_ep_path(workspace.workspace_path)}]")


# Helper functions
def show_text(text, lang=None):
    """显示文本代码块"""
    if lang:
        st.code(text, language=lang, wrap_lines=True)
    elif "\n" in text:
        st.code(text, language="python", wrap_lines=True)
    else:
        st.code(text, language="html", wrap_lines=True)


def highlight_prompts_uri(uri):
    """高亮 URI 的格式"""
    parts = uri.split(":")
    if len(parts) > 1:
        return f"**{parts[0]}:**:green[**{parts[1]}**]"
    return f"**{uri}**"


def llm_log_win(llm_d: list):
    for d in llm_d:
        if "debug_tpl" in d["tag"]:
            uri = d["obj"]["uri"]
            tpl = d["obj"]["template"]
            cxt = d["obj"]["context"]
            rd = d["obj"]["rendered"]
            with st.expander(highlight_prompts_uri(uri), expanded=False, icon="⚙️"):
                t1, t2, t3 = st.tabs([":green[**Rendered**]", ":blue[**Template**]", ":orange[**Context**]"])
                with t1:
                    show_text(rd)
                with t2:
                    show_text(tpl, lang="django")
                with t3:
                    st.json(cxt)
        elif "debug_llm" in d["tag"]:
            system = d["obj"].get("system", None)
            user = d["obj"]["user"]
            resp = d["obj"]["resp"]
            with st.expander(f"**LLM**", expanded=False, icon="🤖"):
                t1, t2, t3 = st.tabs([":green[**Response**]", ":blue[**User**]", ":orange[**System**]"])
                with t1:
                    try:
                        rdict = json.loads(resp)
                        if "code" in rdict:
                            code = rdict["code"]
                            st.markdown(":red[**Code in response dict:**]")
                            st.code(code, language="python", wrap_lines=True, line_numbers=True)
                            rdict.pop("code")
                        elif "spec" in rdict:
                            spec = rdict["spec"]
                            st.markdown(":red[**Spec in response dict:**]")
                            st.markdown(spec)
                            rdict.pop("spec")
                        else:
                            showed_keys = []
                            for k, v in rdict.items():
                                if k.endswith(".py"):
                                    st.markdown(f":red[**{k}**]")
                                    st.code(v, language="python", wrap_lines=True, line_numbers=True)
                                    showed_keys.append(k)
                            for k in showed_keys:
                                rdict.pop(k)
                        st.write(":red[**Other parts (except for the code or spec) in response dict:**]")
                        st.json(rdict)
                    except:
                        try:
                            st.json(resp)
                        except:
                            show_text(resp)
                with t2:
                    show_text(user)
                with t3:
                    show_text(system or "No system prompt available")


def hypothesis_win(hypo):
    try:
        st.code(str(hypo).replace("\n", "\n\n"), wrap_lines=True)
    except Exception as e:
        st.write(hypo.__dict__)


def exp_gen_win(exp_gen_data, llm_data=None):
    st.header("Exp Gen", divider="blue", anchor="exp-gen")
    if state.show_llm_log and llm_data is not None:
        llm_log_win(llm_data["no_tag"])
    st.subheader("Hypothesis")
    hypothesis_win(exp_gen_data["no_tag"].hypothesis)

    st.subheader("pending_tasks")
    for tasks in exp_gen_data["no_tag"].pending_tasks_list:
        task_win(tasks[0])
    st.subheader("Exp Workspace")
    workspace_win(exp_gen_data["no_tag"].experiment_workspace)


def evolving_win(data, key, llm_data=None, base_workspace=None):
    with st.container(border=True):
        if len(data) > 1:
            evo_id = st.slider("Evolving", 0, len(data) - 1, 0, key=key)
        elif len(data) == 1:
            evo_id = 0
        else:
            st.markdown("No evolving.")
            return

        if evo_id in data:
            if state.show_llm_log and llm_data is not None:
                llm_log_win(llm_data[evo_id])
            if data[evo_id]["evolving code"][0] is not None:
                st.subheader("codes")
                workspace_win(
                    data[evo_id]["evolving code"][0],
                    cmp_workspace=data[evo_id - 1]["evolving code"][0] if evo_id > 0 else base_workspace,
                    cmp_name="last evolving code" if evo_id > 0 else "base workspace",
                )
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


def coding_win(data, base_exp, llm_data: dict | None = None):
    st.header("Coding", divider="blue", anchor="coding")
    if llm_data is not None:
        common_llm_data = llm_data.pop("no_tag", [])
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
            evolving_win(
                task_data,
                key=task,
                llm_data=llm_data if llm_data else None,
                base_workspace=base_exp.experiment_workspace,
            )
    else:
        # 旧版未存Task tag的Trace
        evolving_win(
            evolving_data,
            key="coding",
            llm_data=llm_data if llm_data else None,
            base_workspace=base_exp.experiment_workspace,
        )
    if state.show_llm_log:
        llm_log_win(common_llm_data)
    if "no_tag" in data:
        st.subheader("Exp Workspace (coding final)")
        workspace_win(data["no_tag"].experiment_workspace)


def running_win(data, base_exp, llm_data=None, sota_exp=None):
    st.header("Running", divider="blue", anchor="running")
    if llm_data is not None:
        common_llm_data = llm_data.pop("no_tag", [])
    evolving_win(
        {k: v for k, v in data.items() if isinstance(k, int)},
        key="running",
        llm_data=llm_data if llm_data else None,
        base_workspace=base_exp.experiment_workspace,
    )
    if state.show_llm_log and llm_data is not None:
        llm_log_win(common_llm_data)
    if "no_tag" in data:
        st.subheader("Exp Workspace (running final)")
        workspace_win(
            data["no_tag"].experiment_workspace,
            cmp_workspace=sota_exp.experiment_workspace if sota_exp else None,
            cmp_name="last SOTA",
        )
        st.subheader("Result")
        st.write(data["no_tag"].result)
        mle_score_text = data.get("mle_score", "no submission to score")
        mle_score = extract_json(mle_score_text)
        st.subheader(
            "MLE Submission Score"
            + ("✅" if (isinstance(mle_score, dict) and mle_score["score"] is not None) else "❌")
        )
        if isinstance(mle_score, dict):
            st.json(mle_score)
        else:
            st.code(mle_score_text, wrap_lines=True)


def feedback_win(fb_data, llm_data=None):
    fb_data = fb_data["no_tag"]
    st.header("Feedback" + ("✅" if bool(fb_data) else "❌"), divider="orange", anchor="feedback")
    if state.show_llm_log and llm_data is not None:
        llm_log_win(llm_data["no_tag"])
    try:
        st.code(str(fb_data).replace("\n", "\n\n"), wrap_lines=True)
    except Exception as e:
        st.write(fb_data.__dict__)
    if fb_data.exception is not None:
        st.markdown(f"**:red[Exception]**: {fb_data.exception}")


def sota_win(sota_exp, trace):
    st.header("SOTA Experiment", divider="rainbow", anchor="sota-exp")
    if hasattr(trace, "sota_exp_to_submit") and trace.sota_exp_to_submit is not None:
        st.markdown(":orange[trace.**sota_exp_to_submit**]")
        sota_exp = trace.sota_exp_to_submit
    else:
        st.markdown(":orange[trace.**sota_experiment()**]")

    if sota_exp:
        st.markdown(f"**SOTA Exp Hypothesis**")
        hypothesis_win(sota_exp.hypothesis)
        st.markdown("**Exp Workspace**")
        workspace_win(sota_exp.experiment_workspace)
    else:
        st.markdown("No SOTA experiment.")


def main_win(loop_id, llm_data=None):
    loop_data = state.data[loop_id]
    exp_gen_win(loop_data["direct_exp_gen"], llm_data["direct_exp_gen"] if llm_data else None)
    if "coding" in loop_data:
        coding_win(
            loop_data["coding"],
            base_exp=loop_data["direct_exp_gen"]["no_tag"],
            llm_data=llm_data["coding"] if llm_data else None,
        )
    if "running" in loop_data:
        running_win(
            loop_data["running"],
            base_exp=loop_data["coding"]["no_tag"],
            llm_data=llm_data["running"] if llm_data else None,
            sota_exp=(
                state.data[loop_id - 1].get("record", {}).get("SOTA experiment", None)
                if (loop_id - 1) in state.data
                else None
            ),
        )
    if "feedback" in loop_data:
        feedback_win(loop_data["feedback"], llm_data.get("feedback", None) if llm_data else None)
    if "record" in loop_data and "SOTA experiment" in loop_data["record"]:
        sota_win(loop_data["record"]["SOTA experiment"], loop_data["record"]["trace"])


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
    with st.container(border=True):
        df = pd.DataFrame(
            columns=[
                "Component",
                "Hypothesis",
                "Reason",
                "Others",
                "Running Score (valid)",
                "Running Score (test)",
                "Feedback",
                "e-loops(coding)",
                "Time",
                "Exp Gen",
                "Coding",
                "Running",
                "Start Time (UTC+8)",
                "End Time (UTC+8)",
            ],
            index=range(len(state.data) - 1),
        )

        min_id, max_id = get_state_data_range(state.data)
        for loop in range(min_id, max_id + 1):
            loop_data = state.data[loop]
            df.loc[loop, "Component"] = loop_data["direct_exp_gen"]["no_tag"].hypothesis.component
            df.loc[loop, "Hypothesis"] = loop_data["direct_exp_gen"]["no_tag"].hypothesis.hypothesis
            df.loc[loop, "Reason"] = loop_data["direct_exp_gen"]["no_tag"].hypothesis.reason
            df.at[loop, "Others"] = {
                k: v
                for k, v in loop_data["direct_exp_gen"]["no_tag"].hypothesis.__dict__.items()
                if k not in ["component", "hypothesis", "reason"]
            }
            if loop in state.times and state.times[loop]:
                df.loc[loop, "Time"] = str(sum((i.end - i.start for i in state.times[loop]), timedelta())).split(".")[0]
                exp_gen_time = state.times[loop][0].end - state.times[loop][0].start
                df.loc[loop, "Exp Gen"] = str(exp_gen_time).split(".")[0]
                if len(state.times[loop]) > 1:
                    coding_time = state.times[loop][1].end - state.times[loop][1].start
                    df.loc[loop, "Coding"] = str(coding_time).split(".")[0]
                if len(state.times[loop]) > 2:
                    running_time = state.times[loop][2].end - state.times[loop][2].start
                    df.loc[loop, "Running"] = str(running_time).split(".")[0]
                df.loc[loop, "Start Time (UTC+8)"] = state.times[loop][0].start + timedelta(hours=8)
                df.loc[loop, "End Time (UTC+8)"] = state.times[loop][-1].end + timedelta(hours=8)
            if "running" in loop_data and "no_tag" in loop_data["running"]:
                try:
                    df.loc[loop, "Running Score (valid)"] = str(
                        round(loop_data["running"]["no_tag"].result.loc["ensemble"].iloc[0], 5)
                    )
                except:
                    df.loc[loop, "Running Score (valid)"] = "❌"
                if "mle_score" not in state.data[loop]:
                    if "mle_score" in loop_data["running"]:
                        mle_score_txt = loop_data["running"]["mle_score"]
                        state.data[loop]["mle_score"] = extract_json(mle_score_txt)
                        if (
                            state.data[loop]["mle_score"] is not None
                            and state.data[loop]["mle_score"]["score"] is not None
                        ):
                            df.loc[loop, "Running Score (test)"] = str(state.data[loop]["mle_score"]["score"])
                        else:
                            state.data[loop]["mle_score"] = mle_score_txt
                            df.loc[loop, "Running Score (test)"] = "❌"
                    else:
                        mle_score_path = (
                            replace_ep_path(loop_data["running"]["no_tag"].experiment_workspace.workspace_path)
                            / "mle_score.txt"
                        )
                        try:
                            mle_score_txt = mle_score_path.read_text()
                            state.data[loop]["mle_score"] = extract_json(mle_score_txt)
                            if state.data[loop]["mle_score"]["score"] is not None:
                                df.loc[loop, "Running Score (test)"] = str(state.data[loop]["mle_score"]["score"])
                            else:
                                state.data[loop]["mle_score"] = mle_score_txt
                                df.loc[loop, "Running Score (test)"] = "❌"
                        except Exception as e:
                            state.data[loop]["mle_score"] = str(e)
                            df.loc[loop, "Running Score (test)"] = "❌"
                else:
                    if isinstance(state.data[loop]["mle_score"], dict):
                        df.loc[loop, "Running Score (test)"] = str(state.data[loop]["mle_score"]["score"])
                    else:
                        df.loc[loop, "Running Score (test)"] = "❌"

            else:
                df.loc[loop, "Running Score (valid)"] = "N/A"
                df.loc[loop, "Running Score (test)"] = "N/A"

            if "coding" in loop_data:
                if len([i for i in loop_data["coding"].keys() if isinstance(i, int)]) == 0:
                    df.loc[loop, "e-loops(coding)"] = 0
                else:
                    df.loc[loop, "e-loops(coding)"] = (
                        max(i for i in loop_data["coding"].keys() if isinstance(i, int)) + 1
                    )
            if "feedback" in loop_data:
                df.loc[loop, "Feedback"] = "✅" if bool(loop_data["feedback"]["no_tag"]) else "❌"
            else:
                df.loc[loop, "Feedback"] = "N/A"
        st.dataframe(df[df.columns[~df.columns.isin(["Hypothesis", "Reason", "Others"])]])
        st.markdown("### Hypotheses Table")
        st.dataframe(
            df.iloc[:, :8],
            row_height=100,
            column_config={
                "Others": st.column_config.JsonColumn(width="medium"),
                "Reason": st.column_config.TextColumn(width="medium"),
                "Hypothesis": st.column_config.TextColumn(width="large"),
            },
        )

        def comp_stat_func(x: pd.DataFrame):
            total_num = x.shape[0]
            valid_num = x[x["Running Score (test)"] != "N/A"].shape[0]
            success_num = x[x["Feedback"] == "✅"].shape[0]
            avg_e_loops = x["e-loops(coding)"].mean()
            return pd.Series(
                {
                    "Loop Num": total_num,
                    "Valid Loop": valid_num,
                    "Success Loop": success_num,
                    "Valid Rate": round(valid_num / total_num * 100, 2),
                    "Success Rate": round(success_num / total_num * 100, 2),
                    "Avg e-loops(coding)": round(avg_e_loops, 2),
                }
            )

        st1, st2 = st.columns([1, 1])

        # component statistics
        comp_df = (
            df.loc[:, ["Component", "Running Score (test)", "Feedback", "e-loops(coding)"]]
            .groupby("Component")
            .apply(comp_stat_func)
        )
        comp_df.loc["Total"] = comp_df.sum()
        comp_df.loc["Total", "Valid Rate"] = round(
            comp_df.loc["Total", "Valid Loop"] / comp_df.loc["Total", "Loop Num"] * 100, 2
        )
        comp_df.loc["Total", "Success Rate"] = round(
            comp_df.loc["Total", "Success Loop"] / comp_df.loc["Total", "Loop Num"] * 100, 2
        )
        comp_df["Valid Rate"] = comp_df["Valid Rate"].apply(lambda x: f"{x}%")
        comp_df["Success Rate"] = comp_df["Success Rate"].apply(lambda x: f"{x}%")
        comp_df.loc["Total", "Avg e-loops(coding)"] = round(df["e-loops(coding)"].mean(), 2)
        st2.markdown("### Component Statistics")
        st2.dataframe(comp_df)

        # component time statistics
        time_df = df.loc[:, ["Component", "Time", "Exp Gen", "Coding", "Running"]]
        time_df = time_df.astype(
            {
                "Time": "timedelta64[ns]",
                "Exp Gen": "timedelta64[ns]",
                "Coding": "timedelta64[ns]",
                "Running": "timedelta64[ns]",
            }
        )
        st1.markdown("### Time Statistics")
        time_stat_df = time_df.groupby("Component").sum()
        time_stat_df.loc["Total"] = time_stat_df.sum()
        time_stat_df.loc[:, "Exp Gen(%)"] = time_stat_df["Exp Gen"] / time_stat_df["Time"] * 100
        time_stat_df.loc[:, "Coding(%)"] = time_stat_df["Coding"] / time_stat_df["Time"] * 100
        time_stat_df.loc[:, "Running(%)"] = time_stat_df["Running"] / time_stat_df["Time"] * 100
        time_stat_df = time_stat_df.map(lambda x: str(x).split(".")[0] if pd.notnull(x) else "0:00:00")
        st1.dataframe(time_stat_df)


def stdout_win(loop_id: int):
    stdout = load_stdout(state.log_folder / f"{state.log_path}.stdout")
    if stdout.startswith("Please Set"):
        st.toast(stdout, icon="🟡")
        return
    start_index = stdout.find(f"Start Loop {loop_id}")
    end_index = stdout.find(f"Start Loop {loop_id + 1}")
    loop_stdout = LogColors.remove_ansi_codes(stdout[start_index:end_index])
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


def get_folders_sorted(log_path, sort_by_time=False):
    """
    Cache and return the sorted list of folders, with progress printing.
    :param log_path: Log path
    :param sort_by_time: Whether to sort by time, default False (sort by name)
    """
    if not log_path.exists():
        st.toast(f"Path {log_path} does not exist!")
        return []
    with st.spinner("Loading folder list..."):
        folders = [folder for folder in log_path.iterdir() if is_valid_session(folder)]
        if sort_by_time:
            folders = sorted(folders, key=lambda folder: folder.stat().st_mtime, reverse=True)
        else:
            folders = sorted(folders, key=lambda folder: folder.name)
        st.write(f"Found {len(folders)} folders")
    return [folder.name for folder in folders]


# UI - Sidebar
with st.sidebar:
    # TODO: 只是临时的功能
    if any("log.srv" in folder for folder in state.log_folders):
        day_map = {"srv": "最近(srv)", "srv2": "上一批(srv2)", "srv3": "上上批(srv3)"}
        day_srv = st.radio("选择批次", ["srv", "srv2", "srv3"], format_func=lambda x: day_map[x], horizontal=True)
        if day_srv == "srv":
            state.log_folders = [re.sub(r"log\.srv\d*", "log.srv", folder) for folder in state.log_folders]
        elif day_srv == "srv2":
            state.log_folders = [re.sub(r"log\.srv\d*", "log.srv2", folder) for folder in state.log_folders]
        elif day_srv == "srv3":
            state.log_folders = [re.sub(r"log\.srv\d*", "log.srv3", folder) for folder in state.log_folders]

    if "log_folder" in st.query_params:
        state.log_folder = Path(st.query_params["log_folder"])
        state.log_folders = [str(state.log_folder)]
    else:
        state.log_folder = Path(
            st.radio(
                f"Select :blue[**one log folder**]",
                state.log_folders,
                format_func=lambda x: x[x.rfind("amlt") + 5 :].split("/")[0] if "amlt" in x else x,
            )
        )
    if not state.log_folder.exists():
        st.warning(f"Path {state.log_folder} does not exist!")
    else:
        folders = get_folders_sorted(state.log_folder, sort_by_time=False)
        if "selection" in st.query_params:
            default_index = (
                folders.index(st.query_params["selection"]) if st.query_params["selection"] in folders else 0
            )
        else:
            default_index = 0
        state.log_path = st.selectbox(
            f"Select from :blue[**{state.log_folder.absolute()}**]", folders, index=default_index
        )

        if st.button("Refresh Data"):
            if state.log_path is None:
                st.toast("Please select a log path first!", icon="🟡")
                st.stop()

            state.times = load_times(state.log_folder / state.log_path)
            state.data, state.llm_data = load_data(state.log_folder / state.log_path)
            st.rerun()
    st.toggle("**Show LLM Log**", key="show_llm_log")
    st.toggle("*Show stdout*", key="show_stdout")
    st.toggle("*Show save workspace feature*", key="show_save_input")
    st.markdown(
        f"""
- [Exp Gen](#exp-gen)
- [Coding](#coding)
- [Running](#running)
- [Feedback](#feedback)
- [SOTA Experiment](#sota-exp)
"""
    )


def get_state_data_range(state_data):
    # we have a "competition" key in state_data
    # like dict_keys(['competition', 10, 11, 12, 13, 14])
    keys = [k for k in state_data.keys() if isinstance(k, int)]
    return min(keys), max(keys)


# UI - Main
if "competition" in state.data:
    st.title(state.data["competition"])
    st.markdown(f"[share_link](/ds_trace?log_folder={state.log_folder}&selection={state.log_path})")
    summarize_data()
    min_id, max_id = get_state_data_range(state.data)
    if max_id > min_id:
        loop_id = st.slider("Loop", min_id, max_id, min_id)
    else:
        loop_id = min_id
    if state.show_stdout:
        stdout_win(loop_id)
    main_win(loop_id, state.llm_data[loop_id] if loop_id in state.llm_data else None)
