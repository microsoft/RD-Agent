import hashlib
import json
import pickle
import random
import re
from collections import defaultdict
from datetime import time, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from litellm import get_valid_models
from streamlit import session_state as state

from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.log.storage import FileStorage
from rdagent.log.ui.conf import UI_SETTING
from rdagent.log.ui.utils import (
    curve_figure,
    get_sota_exp_stat,
    load_times_info,
    timeline_figure,
    trace_figure,
)
from rdagent.log.utils import (
    LogColors,
    extract_evoid,
    extract_json,
    extract_loopid_func_name,
    is_valid_session,
)
from rdagent.oai.backend.litellm import LITELLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend

# Import necessary classes for the response format
from rdagent.scenarios.data_science.proposal.exp_gen.proposal import (
    CodingSketch,
    HypothesisList,
    ScenarioChallenges,
    TraceChallenges,
)
from rdagent.utils.agent.tpl import T
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
if "sota_info" not in state:
    state.sota_info = None

available_models = get_valid_models()
LITELLM_SETTINGS.dump_chat_cache = False
LITELLM_SETTINGS.dump_embedding_cache = False
LITELLM_SETTINGS.use_chat_cache = False
LITELLM_SETTINGS.use_embedding_cache = False


def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d


def load_data(log_path: Path):
    data = defaultdict(lambda: defaultdict(dict))
    llm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    token_costs = defaultdict(list)

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
        elif "token_cost" in msg.tag:
            token_costs[li].append(msg)
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

    return (
        convert_defaultdict_to_dict(data),
        convert_defaultdict_to_dict(llm_data),
        convert_defaultdict_to_dict(token_costs),
    )


if UI_SETTING.enable_cache:
    load_data = st.cache_data(persist=True)(load_data)


def load_stdout(stdout_path: Path):
    if stdout_path.exists():
        stdout = stdout_path.read_text()
    else:
        stdout = f"Please Set: {stdout_path}"
    return stdout


# UI windows
def task_win(task):
    with st.expander(f"**:violet[{task.name}]**", expanded=False):
        st.markdown(task.description)
        if hasattr(task, "package_info"):
            st.markdown(f"**:blue[Package Info:]**")
            st.code(task.package_info)
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
            with st.popover(f":violet[**Diff with {cmp_name}**]", use_container_width=True, icon="🔍"):
                st.code("".join(diff), language="diff", wrap_lines=True, line_numbers=True)

        rtime = workspace.running_info.running_time
        time_str = timedelta_to_str(timedelta(seconds=rtime) if rtime else None) or "00:00:00"

        with st.popover(
            f"⏱️{time_str} 📂Files in :blue[{replace_ep_path(workspace.workspace_path)}]", use_container_width=True
        ):
            st.write(replace_ep_path(workspace.workspace_path))
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
        st.code(text, language=lang, wrap_lines=True, line_numbers=True)
    elif "\n" in text:
        st.code(text, language="python", wrap_lines=True, line_numbers=True)
    else:
        st.code(text, language="html", wrap_lines=True)


def highlight_prompts_uri(uri):
    """高亮 URI 的格式"""
    parts = uri.split(":")
    if len(parts) > 1:
        return f"**{parts[0]}:**:green[**{parts[1]}**]"
    return f"**{uri}**"


def llm_log_win(llm_d: list):
    def to_str_recursive(obj):
        if isinstance(obj, dict):
            return {k: to_str_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_str_recursive(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(to_str_recursive(v) for v in obj)
        else:
            return str(obj)

    for d in llm_d:
        if "debug_tpl" in d["tag"]:
            uri = d["obj"]["uri"]
            if "filter_redundant_text" in uri:
                continue
            tpl = d["obj"]["template"]
            cxt = d["obj"]["context"]
            rd = d["obj"]["rendered"]
            with st.popover(highlight_prompts_uri(uri), icon="⚙️", use_container_width=True):
                t1, t2, t3 = st.tabs([":green[**Rendered**]", ":blue[**Template**]", ":orange[**Context**]"])
                with t1:
                    show_text(rd)
                with t2:
                    show_text(tpl, lang="django")
                with t3:
                    st.json(to_str_recursive(cxt))
        elif "debug_llm" in d["tag"]:
            system = d["obj"].get("system", None)
            user = d["obj"]["user"]
            resp = d["obj"]["resp"]
            start_time = d["obj"].get("start", "")
            end_time = d["obj"].get("end", "")
            if start_time and end_time:
                start_str = start_time.strftime("%m-%d %H:%M:%S")
                end_str = end_time.strftime("%m-%d %H:%M:%S")
                duration = end_time - start_time
                time_info_str = (
                    f"🕰️:blue[**{start_str} ~ {end_str}**] ⏳:violet[**{round(duration.total_seconds(), 2)}s**]"
                )
            else:
                time_info_str = ""
            with st.expander(f"**LLM** {time_info_str}", icon="🤖", expanded=False):
                t1, t2, t3, t4 = st.tabs(
                    [":green[**Response**]", ":blue[**User**]", ":orange[**System**]", ":violet[**ChatBot**]"]
                )
                with t1:
                    try:
                        rdict = json.loads(resp)
                        showed_keys = []
                        for k, v in rdict.items():
                            if k.endswith(".py") or k.endswith(".md"):
                                st.markdown(f":red[**{k}**]")
                                st.code(v, language="python", wrap_lines=True, line_numbers=True)
                                showed_keys.append(k)
                        for k in showed_keys:
                            rdict.pop(k)
                        if len(showed_keys) > 0:
                            st.write(":red[**Other parts (except for the code or spec) in response dict:**]")
                        st.json(rdict)
                    except:
                        show_text(resp)
                with t2:
                    show_text(user)
                with t3:
                    show_text(system or "No system prompt available")
                with t4:
                    input_c, resp_c = st.columns(2)
                    key = hashlib.md5(resp.encode()).hexdigest()
                    with input_c:
                        btc1, btc2, btc3 = st.columns(3)
                        trace_model = (
                            state.data.get("settings", {})
                            .get("LITELLM_SETTINGS", {})
                            .get("chat_model", available_models[0])
                        )
                        trace_reasoning_effort = (
                            state.data.get("settings", {}).get("LITELLM_SETTINGS", {}).get("reasoning_effort", None)
                        )
                        LITELLM_SETTINGS.chat_model = btc1.selectbox(
                            "Chat Model",
                            options=available_models,
                            index=available_models.index(trace_model),
                            key=key + "_chat_model",
                        )
                        LITELLM_SETTINGS.reasoning_effort = btc2.selectbox(
                            "Reasoning Effort",
                            options=[None, "low", "medium", "high"],
                            index=[None, "low", "medium", "high"].index(trace_reasoning_effort),
                            key=key + "_reasoning_effort",
                        )
                        rf = btc3.selectbox(
                            "Response Format",
                            options=[None, ScenarioChallenges, TraceChallenges, HypothesisList, CodingSketch],
                            format_func=lambda x: x.__name__ if x else "None",
                            key=key + "_response_format",
                        )
                        json_mode = st.checkbox("JSON Mode", value=False, key=key + "_json_mode")
                        sys_p = input_c.text_area(label="system", value=system, height="content", key=key + "_system")
                        user_p = input_c.text_area(label="user", value=user, height="content", key=key + "_user")
                    with resp_c:
                        if st.button("Call LLM", key=key + "_call_llm"):
                            with st.spinner("Calling LLM..."):
                                try:
                                    resp_new = APIBackend().build_messages_and_create_chat_completion(
                                        user_prompt=user_p,
                                        system_prompt=sys_p,
                                        json_mode=json_mode,
                                        response_format=rf,
                                    )
                                except Exception as e:
                                    resp_new = f"Error: {e}"
                            try:  # json format string
                                rdict = json.loads(resp_new)
                                st.json(rdict)
                            except:
                                try:  # common string
                                    st.code(resp_new, wrap_lines=True, line_numbers=True)
                                except:  # response format type
                                    st.write(resp_new)


def hypothesis_win(hypo):
    try:
        st.code(str(hypo).replace("\n", "\n\n"), wrap_lines=True)
    except Exception as e:
        st.write(hypo.__dict__)


def exp_gen_win(exp_gen_data, llm_data=None):
    st.header("Exp Gen", divider="blue", anchor="exp-gen")
    if state.show_llm_log and llm_data is not None:
        llm_log_win(llm_data["no_tag"])
    st.subheader("💡 Hypothesis")
    hypothesis_win(exp_gen_data["no_tag"].hypothesis)

    st.subheader("📋 pending_tasks")
    for tasks in exp_gen_data["no_tag"].pending_tasks_list:
        task_win(tasks[0])
    st.subheader("📁 Exp Workspace")
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

            # get evolving workspace
            if "evolving code" in data[evo_id] and data[evo_id]["evolving code"][0] is not None:
                evolving_code_workspace = data[evo_id]["evolving code"][0]
            else:
                evolving_code_workspace = None

            if evolving_code_workspace is not None:
                st.subheader("codes")
                workspace_win(
                    evolving_code_workspace,
                    cmp_workspace=data[evo_id - 1]["evolving code"][0] if evo_id > 0 else base_workspace,
                    cmp_name="last evolving code" if evo_id > 0 else "base workspace",
                )
                fb = data[evo_id]["evolving feedback"][0]
                st.subheader("evolving feedback" + ("✅" if bool(fb) else "❌"))
                f1, f2, f3, f4 = st.tabs(["execution", "return_checking", "code", "others"])
                other_attributes = {
                    k: v for k, v in fb.__dict__.items() if k not in ["execution", "return_checking", "code"]
                }
                f1.code(fb.execution, wrap_lines=True)
                f2.code(fb.return_checking, wrap_lines=True)
                f3.code(fb.code, wrap_lines=True)
                f4.json(other_attributes)
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


def running_win(data, base_exp, llm_data=None, last_sota_exp=None):
    st.header("Running", divider="blue", anchor="running")
    if llm_data is not None:
        common_llm_data = llm_data.pop("no_tag", [])
    evolving_win(
        {k: v for k, v in data.items() if isinstance(k, int)},
        key="running",
        llm_data=llm_data if llm_data else None,
        base_workspace=base_exp.experiment_workspace if base_exp else None,
    )
    if state.show_llm_log and llm_data is not None:
        llm_log_win(common_llm_data)
    if "no_tag" in data:
        st.subheader("Exp Workspace (running final)")
        workspace_win(
            data["no_tag"].experiment_workspace,
            cmp_workspace=last_sota_exp.experiment_workspace if last_sota_exp else None,
            cmp_name="last SOTA(to_submit)",
        )
        st.subheader("Result")
        try:
            st.write(data["no_tag"].result)
        except AttributeError as e:  # Compatible with old versions
            st.write(data["no_tag"].__dict__["result"])
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
    if "no_tag" not in fb_data:
        st.header("Feedback", divider="orange", anchor="feedback")
        return
    fb = fb_data["no_tag"]
    st.header("Feedback" + ("✅" if bool(fb) else "❌"), divider="orange", anchor="feedback")
    if state.show_llm_log and llm_data is not None:
        llm_log_win(llm_data["no_tag"])
    try:
        st.code(str(fb).replace("\n", "\n\n"), wrap_lines=True)
    except Exception as e:
        st.write(fb.__dict__)
    if fb.exception is not None:
        st.markdown(f"**:red[Exception]**: {fb.exception}")


def sota_win(sota_exp, trace):
    st.subheader("SOTA Experiment", divider="rainbow", anchor="sota-exp")
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
        # get last SOTA_exp_to_submit
        last_sota_exp = None
        if "record" in loop_data:
            current_trace = loop_data["record"]["trace"]
            current_selection = current_trace.get_current_selection()
            if len(current_selection) > 0:  # TODO: Why current_selection can be "()"?
                current_idx = current_selection[0]
                parent_idxs = current_trace.get_parents(current_idx)
                if len(parent_idxs) >= 2 and hasattr(current_trace, "idx2loop_id"):
                    parent_idx = parent_idxs[-2]
                    parent_loop_id = current_trace.idx2loop_id[parent_idx]
                    if parent_loop_id in state.data:
                        # in some cases, the state.data is synthesized, logs does not necessarily exist
                        last_sota_exp = state.data[parent_loop_id]["record"].get("sota_exp_to_submit", None)

        running_win(
            loop_data["running"],
            base_exp=loop_data["coding"].get("no_tag", None),
            llm_data=llm_data["running"] if llm_data else None,
            last_sota_exp=last_sota_exp,
        )
    if "feedback" in loop_data:
        feedback_win(loop_data["feedback"], llm_data.get("feedback", None) if llm_data else None)
    if "record" in loop_data and "SOTA experiment" in loop_data["record"]:
        st.header("Record", divider="violet", anchor="record")
        if state.show_llm_log and llm_data is not None and "record" in llm_data:
            llm_log_win(llm_data["record"]["no_tag"])
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


def get_llm_call_stats(llm_data: dict) -> tuple[int, int]:
    total_llm_call = 0
    total_filter_call = 0
    total_call_duration = timedelta()
    filter_call_duration = timedelta()
    filter_sys_prompt = T("rdagent.utils.prompts:filter_redundant_text.system").r()
    for li, loop_d in llm_data.items():
        for fn, loop_fn_d in loop_d.items():
            for k, v in loop_fn_d.items():
                for d in v:
                    if "debug_llm" in d["tag"]:
                        total_llm_call += 1
                        total_call_duration += d["obj"].get("end", timedelta()) - d["obj"].get("start", timedelta())
                        if "system" in d["obj"] and filter_sys_prompt == d["obj"]["system"]:
                            total_filter_call += 1
                            filter_call_duration += d["obj"].get("end", timedelta()) - d["obj"].get(
                                "start", timedelta()
                            )

    return total_llm_call, total_filter_call, total_call_duration, filter_call_duration


def get_timeout_stats(llm_data: dict):
    timeout_stat = {
        "coding": {
            "total": 0,
            "timeout": 0,
        },
        "running": {
            "total": 0,
            "timeout": 0,
        },
    }
    for li, loop_d in llm_data.items():
        for fn, loop_fn_d in loop_d.items():
            for k, v in loop_fn_d.items():
                for d in v:
                    if "debug_tpl" in d["tag"] and "eval.user" in d["obj"]["uri"] and "stdout" in d["obj"]["context"]:
                        stdout = d["obj"]["context"]["stdout"]
                        if "The running time exceeds" in stdout:  # Timeout case
                            timeout_stat[fn]["timeout"] += 1
                        timeout_stat[fn]["total"] += 1

    return timeout_stat


def timedelta_to_str(td: timedelta | None) -> str:
    if isinstance(td, timedelta):
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return td


def summarize_win():
    st.header("Summary", divider="rainbow")
    with st.container(border=True):
        min_id, max_id = get_state_data_range(state.data)
        info0, info1, info2, info3, info4, info5, info6, info7 = st.columns(8)
        show_trace_dag = info0.toggle("Show trace DAG", key="show_trace_dag")
        only_success = info0.toggle("Only Success", key="only_success")
        with info1.popover("LITELLM", icon="⚙️"):
            st.write(state.data.get("settings", {}).get("LITELLM_SETTINGS", "No settings found."))
        with info2.popover("RD_AGENT", icon="⚙️"):
            st.write(state.data.get("settings", {}).get("RD_AGENT_SETTINGS", "No settings found."))
        with info3.popover("RDLOOP", icon="⚙️"):
            st.write(state.data.get("settings", {}).get("RDLOOP_SETTINGS", "No settings found."))

        llm_call, llm_filter_call, llm_call_duration, filter_call_duration = get_llm_call_stats(state.llm_data)
        info4.metric("LLM Calls", llm_call, help=timedelta_to_str(llm_call_duration))
        info5.metric(
            "LLM Filter Calls",
            llm_filter_call,
            help=timedelta_to_str(filter_call_duration),
        )

        timeout_stats = get_timeout_stats(state.llm_data)
        coding_timeout_pct = (
            round(timeout_stats["coding"]["timeout"] / timeout_stats["coding"]["total"] * 100, 2)
            if timeout_stats["coding"]["total"] > 0
            else 0
        )
        info6.metric(
            "Timeouts (C)",
            f"{coding_timeout_pct}%",
            help=f"{timeout_stats['coding']['timeout']}/{timeout_stats['coding']['total']}",
        )
        running_timeout_pct = (
            round(timeout_stats["running"]["timeout"] / timeout_stats["running"]["total"] * 100, 2)
            if timeout_stats["running"]["total"] > 0
            else 0
        )
        info7.metric(
            "Timeouts (R)",
            f"{running_timeout_pct}%",
            help=f"{timeout_stats['running']['timeout']}/{timeout_stats['running']['total']}",
        )

        final_trace = list(FileStorage(state.log_folder / state.log_path).iter_msg(tag="record.trace"))[-1].content
        if show_trace_dag:
            st.markdown("### Trace DAG")
            merge_loops = []
            for loop_id in state.llm_data.keys():
                if "direct_exp_gen" not in state.llm_data[loop_id]:
                    continue
                if "scenarios.data_science.proposal.exp_gen.merge" in "".join(
                    [i["obj"]["uri"] for i in state.llm_data[loop_id]["direct_exp_gen"]["no_tag"] if "uri" in i["obj"]]
                ):
                    merge_loops.append(loop_id)
            st.pyplot(trace_figure(final_trace, merge_loops))

        # Find all root nodes (for grouping loops by trace)
        root_nodes = {}
        parent_nodes = {}
        for node in range(len(final_trace.hist)):
            parents = final_trace.get_parents(node)
            root_nodes[node] = parents[0]
            parent_nodes[node] = parents[-2] if len(parents) > 1 else None
        if hasattr(final_trace, "idx2loop_id"):
            root_nodes = {final_trace.idx2loop_id[n]: final_trace.idx2loop_id[r] for n, r in root_nodes.items()}
            parent_nodes = {
                final_trace.idx2loop_id[n]: final_trace.idx2loop_id[r] if r is not None else r
                for n, r in parent_nodes.items()
            }

        # Generate Summary Table
        df = pd.DataFrame(
            columns=[
                "Root N",
                "Parent N",
                "Component",
                "Hypothesis",
                "Reason",
                "Others",
                "Run Score (valid)",
                "Run Score (test)",
                "Feedback",
                "e-loops(c)",
                "e-loops(r)",
                "COST($)",
                "Time",
                "Exp Gen",
                "Coding",
                "Running",
            ],
            index=range(min_id, max_id + 1),
        )

        valid_results = {}
        sota_loop_id = state.sota_info[1] if state.sota_info else None
        for loop in range(min_id, max_id + 1):
            loop_data = state.data[loop]
            df.loc[loop, "Parent N"] = parent_nodes.get(loop, None)
            df.loc[loop, "Root N"] = root_nodes.get(loop, None)
            df.loc[loop, "Component"] = loop_data["direct_exp_gen"]["no_tag"].hypothesis.component
            df.loc[loop, "Hypothesis"] = loop_data["direct_exp_gen"]["no_tag"].hypothesis.hypothesis
            df.loc[loop, "Reason"] = loop_data["direct_exp_gen"]["no_tag"].hypothesis.reason
            df.at[loop, "Others"] = {
                k: v
                for k, v in loop_data["direct_exp_gen"]["no_tag"].hypothesis.__dict__.items()
                if k not in ["component", "hypothesis", "reason"] and v is not None
            }
            df.loc[loop, "COST($)"] = sum(tc.content["cost"] for tc in state.token_costs[loop])

            # Time Stats
            exp_gen_time = timedelta()
            coding_time = timedelta()
            running_time = timedelta()
            all_steps_time = timedelta()
            if loop in state.times:
                for step_name, step_time in state.times[loop].items():
                    step_duration = step_time["end_time"] - step_time["start_time"]
                    if step_name == "exp_gen":
                        exp_gen_time += step_duration
                        all_steps_time += step_duration
                    elif step_name == "coding":
                        coding_time += step_duration
                        all_steps_time += step_duration
                    elif step_name == "running":
                        running_time += step_duration
                        all_steps_time += step_duration
                    elif step_name in ["feedback", "record"]:
                        all_steps_time += step_duration
            df.loc[loop, "Time"] = timedelta_to_str(all_steps_time)
            df.loc[loop, "Exp Gen"] = timedelta_to_str(exp_gen_time)
            df.loc[loop, "Coding"] = timedelta_to_str(coding_time)
            df.loc[loop, "Running"] = timedelta_to_str(running_time)

            if "running" in loop_data and "no_tag" in loop_data["running"]:
                try:
                    try:
                        running_result = loop_data["running"]["no_tag"].result
                    except AttributeError as e:  # Compatible with old versions
                        running_result = loop_data["running"]["no_tag"].__dict__["result"]
                    df.loc[loop, "Run Score (valid)"] = str(round(running_result.loc["ensemble"].iloc[0], 5))
                    valid_results[loop] = running_result
                except:
                    df.loc[loop, "Run Score (valid)"] = "❌"
                if "mle_score" not in state.data[loop]:
                    if "mle_score" in loop_data["running"]:
                        mle_score_txt = loop_data["running"]["mle_score"]
                        state.data[loop]["mle_score"] = extract_json(mle_score_txt)
                        if (
                            state.data[loop]["mle_score"] is not None
                            and state.data[loop]["mle_score"]["score"] is not None
                        ):
                            medal_emoji = (
                                "🥇"
                                if state.data[loop]["mle_score"]["gold_medal"]
                                else (
                                    "🥈"
                                    if state.data[loop]["mle_score"]["silver_medal"]
                                    else "🥉" if state.data[loop]["mle_score"]["bronze_medal"] else ""
                                )
                            )
                            df.loc[loop, "Run Score (test)"] = f"{medal_emoji} {state.data[loop]['mle_score']['score']}"
                        else:
                            state.data[loop]["mle_score"] = mle_score_txt
                            df.loc[loop, "Run Score (test)"] = "❌"
                    else:
                        mle_score_path = (
                            replace_ep_path(loop_data["running"]["no_tag"].experiment_workspace.workspace_path)
                            / "mle_score.txt"
                        )
                        try:
                            mle_score_txt = mle_score_path.read_text()
                            state.data[loop]["mle_score"] = extract_json(mle_score_txt)
                            if state.data[loop]["mle_score"]["score"] is not None:
                                medal_emoji = (
                                    "🥇"
                                    if state.data[loop]["mle_score"]["gold_medal"]
                                    else (
                                        "🥈"
                                        if state.data[loop]["mle_score"]["silver_medal"]
                                        else "🥉" if state.data[loop]["mle_score"]["bronze_medal"] else ""
                                    )
                                )
                                df.loc[loop, "Run Score (test)"] = (
                                    f"{medal_emoji} {state.data[loop]['mle_score']['score']}"
                                )
                            else:
                                state.data[loop]["mle_score"] = mle_score_txt
                                df.loc[loop, "Run Score (test)"] = "❌"
                        except Exception as e:
                            state.data[loop]["mle_score"] = str(e)
                            df.loc[loop, "Run Score (test)"] = "❌"
                else:
                    if isinstance(state.data[loop]["mle_score"], dict):
                        medal_emoji = (
                            "🥇"
                            if state.data[loop]["mle_score"]["gold_medal"]
                            else (
                                "🥈"
                                if state.data[loop]["mle_score"]["silver_medal"]
                                else "🥉" if state.data[loop]["mle_score"]["bronze_medal"] else ""
                            )
                        )
                        df.loc[loop, "Run Score (test)"] = f"{medal_emoji} {state.data[loop]['mle_score']['score']}"
                    else:
                        df.loc[loop, "Run Score (test)"] = "❌"

            else:
                df.loc[loop, "Run Score (valid)"] = "N/A"
                df.loc[loop, "Run Score (test)"] = "N/A"

            if "coding" in loop_data:
                if len([i for i in loop_data["coding"].keys() if isinstance(i, int)]) == 0:
                    df.loc[loop, "e-loops(c)"] = 0
                else:
                    df.loc[loop, "e-loops(c)"] = max(i for i in loop_data["coding"].keys() if isinstance(i, int)) + 1
            if "running" in loop_data:
                if len([i for i in loop_data["running"].keys() if isinstance(i, int)]) == 0:
                    df.loc[loop, "e-loops(r)"] = 0
                else:
                    df.loc[loop, "e-loops(r)"] = max(i for i in loop_data["running"].keys() if isinstance(i, int)) + 1
            if "feedback" in loop_data:
                fb_emoji_str = (
                    "✅" if "no_tag" in loop_data["feedback"] and bool(loop_data["feedback"]["no_tag"]) else "❌"
                )
                if sota_loop_id == loop:
                    fb_emoji_str += " (💖SOTA)"
                df.loc[loop, "Feedback"] = fb_emoji_str
            else:
                df.loc[loop, "Feedback"] = "N/A"

        if only_success:
            df = df[df["Feedback"].str.contains("✅", na=False)]

        # Add color styling based on root_nodes
        def style_dataframe_by_root(df, root_nodes):
            # Create a color map for different root nodes - using colors that work well in both light and dark modes
            unique_roots = list(set(root_nodes.values()))
            colors = [
                "rgba(255, 99, 132, 0.3)",
                "rgba(54, 162, 235, 0.3)",
                "rgba(75, 192, 75, 0.3)",
                "rgba(255, 159, 64, 0.3)",
                "rgba(153, 102, 255, 0.2)",
                "rgba(255, 205, 86, 0.2)",
                "rgba(199, 199, 199, 0.2)",
                "rgba(83, 102, 255, 0.2)",
            ]
            root_color_map = {root: colors[i % len(colors)] for i, root in enumerate(unique_roots)}

            # Create styling function
            def apply_color(row):
                loop_id = row.name
                if loop_id in root_nodes:
                    root_id = root_nodes[loop_id]
                    color = root_color_map.get(root_id, "rgba(128, 128, 128, 0.1)")
                    return [f"background-color: {color}"] * len(row)
                return [""] * len(row)

            return df.style.apply(apply_color, axis=1)

        styled_df = style_dataframe_by_root(
            df[df.columns[~df.columns.isin(["Hypothesis", "Reason", "Others"])]], root_nodes
        )
        st.dataframe(styled_df)

        # timeline figure
        if state.times:
            with st.popover("Timeline", icon="⏱️", use_container_width=True):
                st.plotly_chart(timeline_figure(state.times))

        # scores curve
        vscores = {}
        for k, vs in valid_results.items():
            if not vs.index.is_unique:
                st.warning(f"Loop {k}'s valid scores index are not unique, only the last one will be kept to show.")
                st.write(vs)
            vscores[k] = vs[~vs.index.duplicated(keep="last")].iloc[:, 0]
        if len(vscores) > 0:
            metric_name = list(vscores.values())[0].name
        else:
            metric_name = "None"
        vscores = pd.DataFrame(vscores)
        if "ensemble" in vscores.index:
            ensemble_row = vscores.loc[["ensemble"]]
            vscores = pd.concat([ensemble_row, vscores.drop("ensemble")])
        vscores = vscores.T
        test_scores = df["Run Score (test)"].str.replace(r"[🥇🥈🥉]\s*", "", regex=True)
        vscores["test"] = test_scores
        vscores.index = [f"L{i}" for i in vscores.index]
        vscores.columns.name = metric_name
        with st.popover("Scores Curve", icon="📈", use_container_width=True):
            st.plotly_chart(curve_figure(vscores))

        st.markdown("### Hypotheses Table")
        hypotheses_df = df.iloc[:, :8].copy()
        others_expanded = pd.json_normalize(hypotheses_df["Others"].fillna({}))
        others_expanded.index = hypotheses_df.index

        hypotheses_df = hypotheses_df.drop("Others", axis=1)
        hypotheses_df = hypotheses_df.drop("Parent N", axis=1)
        hypotheses_df = pd.concat([hypotheses_df.iloc[:, :4], others_expanded, hypotheses_df.iloc[:, 4:]], axis=1)

        styled_hypotheses_table = style_dataframe_by_root(hypotheses_df, root_nodes)
        st.dataframe(
            styled_hypotheses_table,
            row_height=100,
            column_config={
                k: st.column_config.TextColumn(
                    k,
                    width=(
                        "small"
                        if k
                        in ["Component", "Root N", "Parent N", "Run Score (valid)", "Run Score (test)", "problem_label"]
                        else "medium"
                    ),
                )
                for k in hypotheses_df.columns
            },
        )

        def comp_stat_func(x: pd.DataFrame):
            total_num = x.shape[0]
            valid_num = x[x["Run Score (test)"] != "N/A"].shape[0]
            success_num = x[x["Feedback"] == "✅"].shape[0]
            avg_e_loops = x["e-loops(c)"].mean()
            return pd.Series(
                {
                    "Loop Num": total_num,
                    "Valid Loop": valid_num,
                    "Success Loop": success_num,
                    "Valid Rate": round(valid_num / total_num * 100, 2),
                    "Success Rate": round(success_num / total_num * 100, 2),
                    "Avg e-loops(c)": round(avg_e_loops, 2),
                }
            )

        st1, st2 = st.columns([1, 1])

        # component statistics
        comp_df = (
            df.loc[:, ["Component", "Run Score (test)", "Feedback", "e-loops(c)"]]
            .groupby("Component")
            .apply(comp_stat_func, include_groups=False)
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
        comp_df.loc["Total", "Avg e-loops(c)"] = round(df["e-loops(c)"].mean(), 2)
        with st2.popover("Component Statistics", icon="📊", use_container_width=True):
            st.dataframe(comp_df)

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
        time_stat_df = time_df.groupby("Component").sum()
        time_stat_df.loc["Total"] = time_stat_df.sum()
        time_stat_df.loc[:, "Exp Gen(%)"] = (time_stat_df["Exp Gen"] / time_stat_df["Time"] * 100).round(2)
        time_stat_df.loc[:, "Coding(%)"] = (time_stat_df["Coding"] / time_stat_df["Time"] * 100).round(2)
        time_stat_df.loc[:, "Running(%)"] = (time_stat_df["Running"] / time_stat_df["Time"] * 100).round(2)
        for col in ["Time", "Exp Gen", "Coding", "Running"]:
            time_stat_df[col] = time_stat_df[col].map(timedelta_to_str)
        with st1.popover("Time Statistics", icon="⏱️", use_container_width=True):
            st.dataframe(time_stat_df)

        # COST curve
        costs = df["COST($)"].astype(float)
        costs.index = [f"L{i}" for i in costs.index]
        cumulative_costs = costs.cumsum()
        with st.popover("COST Curve", icon="💰", use_container_width=True):
            fig = px.line(
                x=costs.index,
                y=[costs.values, cumulative_costs.values],
                labels={"x": "Loop", "value": "COST($)"},
                title="COST($) per Loop & Cumulative COST($)",
                markers=True,
            )
            fig.update_traces(mode="lines+markers")
            fig.data[0].name = "COST($) per Loop"
            fig.data[1].name = "Cumulative COST($)"
            st.plotly_chart(fig)


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

            state.times = load_times_info(state.log_folder / state.log_path)
            state.data, state.llm_data, state.token_costs = load_data(state.log_folder / state.log_path)
            state.sota_info = get_sota_exp_stat(Path(state.log_folder) / state.log_path, selector="auto")
            st.rerun()
    st.toggle("**Show LLM Log**", key="show_llm_log")
    st.toggle("*Show stdout*", key="show_stdout")
    st.toggle("*Show save workspace*", key="show_save_input")
    st.markdown(
        f"""
- [Summary](#summary)
- [Exp Gen](#exp-gen)
- [Coding](#coding)
- [Running](#running)
- [Feedback](#feedback)
- [Record](#record)
    - [SOTA Experiment](#sota-exp)
"""
    )


def get_state_data_range(state_data):
    # we have a "competition" key in state_data
    # like dict_keys(['competition', 10, 11, 12, 13, 14])
    keys = [
        k
        for k in state_data.keys()
        if isinstance(k, int) and "direct_exp_gen" in state_data[k] and "no_tag" in state_data[k]["direct_exp_gen"]
    ]
    return min(keys), max(keys)


# UI - Main
if "competition" in state.data:
    st.title(
        state.data["competition"]
        + f" ([share_link](/ds_trace?log_folder={state.log_folder}&selection={state.log_path}))"
    )
    summarize_win()
    min_id, max_id = get_state_data_range(state.data)
    if max_id > min_id:
        loop_id = st.slider("Loop", min_id, max_id, min_id)
    else:
        loop_id = min_id
    if state.show_stdout:
        stdout_win(loop_id)
    main_win(loop_id, state.llm_data[loop_id] if loop_id in state.llm_data else None)
