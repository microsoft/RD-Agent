import argparse
import json
import pickle
import re
import time
from pathlib import Path

import streamlit as st
from streamlit import session_state

from rdagent.log.ui.conf import UI_SETTING
from rdagent.log.utils import extract_evoid, extract_loopid_func_name

st.set_page_config(layout="wide", page_title="debug_llm", page_icon="🎓", initial_sidebar_state="expanded")

# 获取 log_path 参数
parser = argparse.ArgumentParser(description="RD-Agent Streamlit App")
parser.add_argument("--log_dir", type=str, help="Path to the log directory")
args = parser.parse_args()


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


if UI_SETTING.enable_cache:
    get_folders_sorted = st.cache_data(get_folders_sorted)


# 设置主日志路径
main_log_path = Path(args.log_dir) if args.log_dir else Path("./log")
if not main_log_path.exists():
    st.error(f"Log dir {main_log_path} does not exist!")
    st.stop()

if "data" not in session_state:
    session_state.data = []
if "log_path" not in session_state:
    session_state.log_path = None

tlist = []


def load_data():
    """加载数据到 session_state 并显示进度"""
    log_file = main_log_path / session_state.log_path / "debug_llm.pkl"
    try:
        with st.spinner(f"正在加载数据文件 {log_file}..."):
            start_time = time.time()
            with open(log_file, "rb") as f:
                session_state.data = pickle.load(f)
            st.success(f"数据加载完成！耗时 {time.time() - start_time:.2f} 秒")
            st.session_state["current_loop"] = 1
    except Exception as e:
        session_state.data = [{"error": str(e)}]
        st.error(f"加载数据失败: {e}")


# UI - Sidebar
with st.sidebar:
    st.markdown(":blue[**Log Path**]")
    manually = st.toggle("Manual Input")
    if manually:
        st.text_input("log path", key="log_path", label_visibility="collapsed")
    else:
        folders = get_folders_sorted(main_log_path)
        st.selectbox(f"**Select from {main_log_path.absolute()}**", folders, key="log_path")

    if st.button("Refresh Data"):
        load_data()
        st.rerun()


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
    return f"**{parts[0]}:**:green[**{parts[1]}**]"


# Display Data
progress_text = st.empty()
progress_bar = st.progress(0)

# 每页展示一个 Loop
LOOPS_PER_PAGE = 1

# 获取所有的 Loop ID
loop_groups = {}
for i, d in enumerate(session_state.data):
    tag = d["tag"]
    loop_id, _ = extract_loopid_func_name(tag)
    if loop_id:
        if loop_id not in loop_groups:
            loop_groups[loop_id] = []
        loop_groups[loop_id].append(d)

# 按 Loop ID 排序
sorted_loop_ids = sorted(loop_groups.keys(), key=int)  # 假设 Loop ID 是数字
total_loops = len(sorted_loop_ids)
total_pages = total_loops  # 每页展示一个 Loop


# simple display
# FIXME: Delete this simple UI if trace have tag(evo_id & loop_id)
# with st.sidebar:
#     start = int(st.text_input("start", 0))
#     end = int(st.text_input("end", 100))
# for m in session_state.data[start:end]:
#     if "tpl" in m["tag"]:
#         obj = m["obj"]
#         uri = obj["uri"]
#         tpl = obj["template"]
#         cxt = obj["context"]
#         rd = obj["rendered"]
#         with st.expander(highlight_prompts_uri(uri), expanded=False, icon="⚙️"):
#             t1, t2, t3 = st.tabs([":green[**Rendered**]", ":blue[**Template**]", ":orange[**Context**]"])
#             with t1:
#                 show_text(rd)
#             with t2:
#                 show_text(tpl, lang="django")
#             with t3:
#                 st.json(cxt)
#     if "llm" in m["tag"]:
#         obj = m["obj"]
#         system = obj.get("system", None)
#         user = obj["user"]
#         resp = obj["resp"]
#         with st.expander(f"**LLM**", expanded=False, icon="🤖"):
#             t1, t2, t3 = st.tabs([":green[**Response**]", ":blue[**User**]", ":orange[**System**]"])
#             with t1:
#                 try:
#                     rdict = json.loads(resp)
#                     if "code" in rdict:
#                         code = rdict["code"]
#                         st.markdown(":red[**Code in response dict:**]")
#                         st.code(code, language="python", wrap_lines=True, line_numbers=True)
#                         rdict.pop("code")
#                     elif "spec" in rdict:
#                         spec = rdict["spec"]
#                         st.markdown(":red[**Spec in response dict:**]")
#                         st.markdown(spec)
#                         rdict.pop("spec")
#                     else:
#                         # show model codes
#                         showed_keys = []
#                         for k, v in rdict.items():
#                             if k.startswith("model_") and k.endswith(".py"):
#                                 st.markdown(f":red[**{k}**]")
#                                 st.code(v, language="python", wrap_lines=True, line_numbers=True)
#                                 showed_keys.append(k)
#                         for k in showed_keys:
#                             rdict.pop(k)
#                     st.write(":red[**Other parts (except for the code or spec) in response dict:**]")
#                     st.json(rdict)
#                 except:
#                     st.json(resp)
#             with t2:
#                 show_text(user)
#             with t3:
#                 show_text(system or "No system prompt available")


if total_pages:
    # 初始化 current_loop
    if "current_loop" not in st.session_state:
        st.session_state["current_loop"] = 1

    # Loop 导航按钮
    col1, col2, col3, col4, col5 = st.sidebar.columns([1.2, 1, 2, 1, 1.2])

    with col1:
        if st.button("|<"):  # 首页
            st.session_state["current_loop"] = 1
    with col2:
        if st.button("<") and st.session_state["current_loop"] > 1:  # 上一页
            st.session_state["current_loop"] -= 1
    with col3:
        # 下拉列表显示所有 Loop
        st.session_state["current_loop"] = st.selectbox(
            "选择 Loop",
            options=list(range(1, total_loops + 1)),
            index=st.session_state["current_loop"] - 1,  # 默认选中当前 Loop
            label_visibility="collapsed",  # 隐藏标签
        )
    with col4:
        if st.button("\>") and st.session_state["current_loop"] < total_loops:  # 下一页
            st.session_state["current_loop"] += 1
    with col5:
        if st.button("\>|"):  # 最后一页
            st.session_state["current_loop"] = total_loops

    # 获取当前 Loop
    current_loop = st.session_state["current_loop"]

    # 渲染当前 Loop 数据
    loop_id = sorted_loop_ids[current_loop - 1]
    progress_text = st.empty()
    progress_text.text(f"正在处理 Loop {loop_id}...")
    progress_bar.progress(current_loop / total_loops, text=f"Loop :green[**{current_loop}**] / {total_loops}")

    # 渲染 Loop Header
    loop_anchor = f"Loop_{loop_id}"
    if loop_anchor not in tlist:
        tlist.append(loop_anchor)
        st.header(loop_anchor, anchor=loop_anchor, divider="blue")

    # 渲染当前 Loop 的所有数据
    loop_data = loop_groups[loop_id]
    for d in loop_data:
        tag = d["tag"]
        obj = d["obj"]
        _, func_name = extract_loopid_func_name(tag)
        evo_id = extract_evoid(tag)

        func_anchor = f"loop_{loop_id}.{func_name}"
        if func_anchor not in tlist:
            tlist.append(func_anchor)
            st.header(f"in *{func_name}*", anchor=func_anchor, divider="green")

        evo_anchor = f"loop_{loop_id}.evo_step_{evo_id}"
        if evo_id and evo_anchor not in tlist:
            tlist.append(evo_anchor)
            st.subheader(f"evo_step_{evo_id}", anchor=evo_anchor, divider="orange")

        # 根据 tag 渲染内容
        if "debug_exp_gen" in tag:
            with st.expander(
                f"Exp in :violet[**{obj.experiment_workspace.workspace_path}**]", expanded=False, icon="🧩"
            ):
                st.write(obj)
        elif "debug_tpl" in tag:
            uri = obj["uri"]
            tpl = obj["template"]
            cxt = obj["context"]
            rd = obj["rendered"]
            with st.expander(highlight_prompts_uri(uri), expanded=False, icon="⚙️"):
                t1, t2, t3 = st.tabs([":green[**Rendered**]", ":blue[**Template**]", ":orange[**Context**]"])
                with t1:
                    show_text(rd)
                with t2:
                    show_text(tpl, lang="django")
                with t3:
                    st.json(cxt)
        elif "debug_llm" in tag:
            system = obj.get("system", None)
            user = obj["user"]
            resp = obj["resp"]
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
                            # show model codes
                            showed_keys = []
                            for k, v in rdict.items():
                                if k.startswith("model_") and k.endswith(".py"):
                                    st.markdown(f":red[**{k}**]")
                                    st.code(v, language="python", wrap_lines=True, line_numbers=True)
                                    showed_keys.append(k)
                            for k in showed_keys:
                                rdict.pop(k)
                        st.write(":red[**Other parts (except for the code or spec) in response dict:**]")
                        st.json(rdict)
                    except:
                        st.json(resp)
                with t2:
                    show_text(user)
                with t3:
                    show_text(system or "No system prompt available")

    progress_text.text("当前 Loop 数据处理完成！")

    # Sidebar TOC
    with st.sidebar:
        toc = "\n".join([f"- [{t}](#{t})" if t.startswith("L") else f"  - [{t.split('.')[1]}](#{t})" for t in tlist])
        st.markdown(toc, unsafe_allow_html=True)
