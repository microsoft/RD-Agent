import pickle
import re
from collections import defaultdict
from pathlib import Path

import streamlit as st
from streamlit import session_state as state

from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.log.mle_summary import is_valid_session
from rdagent.log.storage import FileStorage
from rdagent.log.ui.conf import UI_SETTING

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
if "show_llm_log" not in state:
    state.show_llm_log = False
if "data" not in state:
    state.data = defaultdict(lambda: defaultdict(dict))
if "llm_data" not in state:
    state.llm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

summary_page = st.Page("ds_summary.py", title="Summary", icon="📊")
trace_page = st.Page("ds_trace.py", title="Trace", icon="📈")
st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")
st.navigation([summary_page, trace_page]).run()


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


def extract_loopid_func_name(tag):
    """提取 Loop ID 和函数名称"""
    match = re.search(r"Loop_(\d+)\.([^.]+)", tag)
    return match.groups() if match else (None, None)


def extract_evoid(tag):
    """提取 EVO ID"""
    match = re.search(r"\.evo_loop_(\d+)\.", tag)
    return match.group(1) if match else None


def load_times(log_path: Path):
    """加载时间数据"""
    state.times = defaultdict(lambda: defaultdict(dict))
    for msg in FileStorage(log_path).iter_msg():
        if msg.tag and "llm" not in msg.tag and "session" not in msg.tag:
            li, fn = extract_loopid_func_name(msg.tag)
            if li:
                li = int(li)

            # read times
            loop_obj_path = log_path / "__session__" / f"{li}" / "4_record"
            if loop_obj_path.exists():
                try:
                    state.times[li] = DataScienceRDLoop.load(loop_obj_path, do_truncate=False).loop_trace[li]
                except Exception as e:
                    pass


def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d


@st.cache_data
def load_data(log_path: Path):
    data = defaultdict(lambda: defaultdict(dict))
    for msg in FileStorage(log_path).iter_msg():
        if msg.tag and "llm" not in msg.tag and "session" not in msg.tag:
            if msg.tag == "competition":
                data["competition"] = msg.content
                continue

            li, fn = extract_loopid_func_name(msg.tag)
            if li:
                li = int(li)

            ei = extract_evoid(msg.tag)
            msg.tag = re.sub(r"\.evo_loop_\d+", "", msg.tag)
            msg.tag = re.sub(r"Loop_\d+\.[^.]+\.?", "", msg.tag)
            msg.tag = msg.tag.strip()

            if ei:
                if int(ei) not in data[li][fn]:
                    data[li][fn][int(ei)] = {}
                data[li][fn][int(ei)][msg.tag] = msg.content
            else:
                if msg.tag:
                    data[li][fn][msg.tag] = msg.content
                else:
                    if not isinstance(msg.content, str):
                        data[li][fn]["no_tag"] = msg.content

    # debug_llm data
    llm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    llm_log_p = log_path / "debug_llm.pkl"
    with st.spinner("正在加载 debug_llm.pkl..."):
        rd = pickle.loads(llm_log_p.read_bytes())
        for i, d in enumerate(rd):
            t = d["tag"]
            if "debug_exp_gen" in t:
                continue
            if "debug_tpl" in t and "filter_" in d["obj"]["uri"]:
                continue
            lid, fn = extract_loopid_func_name(t)
            ei = extract_evoid(t)
            if lid:
                lid = int(lid)
            if ei:
                ei = int(ei)

            if ei is not None:
                llm_data[lid][fn][ei].append(d)
            else:
                llm_data[lid][fn]["no_tag"].append(d)

    return convert_defaultdict_to_dict(data), convert_defaultdict_to_dict(llm_data)


# UI - Sidebar
with st.sidebar:
    st.subheader("Pages", divider="rainbow")
    st.page_link(summary_page, icon="📊")
    st.page_link(trace_page, icon="📈")

    st.subheader("Settings", divider="rainbow")
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

            state.data, state.llm_data = load_data(state.log_folder / state.log_path)
            load_times(state.log_folder / state.log_path)
            st.rerun()
    st.toggle("Show LLM Log", key="show_llm_log")
    st.toggle("Show stdout", key="show_stdout")
    st.markdown(
        f"""
- [Exp Gen](#exp-gen)
- [Coding](#coding)
- [Running](#running)
- [Feedback](#feedback)
- [SOTA Experiment](#sota-exp)
"""
    )
