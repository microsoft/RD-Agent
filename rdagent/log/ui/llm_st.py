import re
import argparse
import json
from pathlib import Path

import streamlit as st
from streamlit import session_state

st.set_page_config(layout="wide", page_title="debug_llm", page_icon="🎓", initial_sidebar_state="expanded")

# 获取log_path参数
parser = argparse.ArgumentParser(description="RD-Agent Streamlit App")
parser.add_argument("--log_dir", type=str, help="Path to the log directory")
args = parser.parse_args()
if args.log_dir:
    main_log_path = Path(args.log_dir)
    if not main_log_path.exists():
        st.error(f"Log dir `{main_log_path}` does not exist!")
        st.stop()
else:
    main_log_path = Path("./log")

if "data" not in session_state:
    session_state.data = []
if "log_path" not in session_state:
    session_state.log_path = None

tlist = []

def load_data():
    try:
        with open(f"{main_log_path}/{session_state.log_path}/debug_llm.json", "r") as f:
            session_state.data = json.load(f)
    except Exception as e:
        session_state.data = [{"error": str(e)}]


# Add a button to refresh the data
with st.sidebar:
    lc1, lc2 = st.columns([1, 2], vertical_alignment="center")
    with lc1:
        st.markdown(":blue[**Log Path**]")
    with lc2:
        manually = st.toggle("Manual Input")
    if manually:
        st.text_input("log path", key="log_path", label_visibility="collapsed")
    else:
        folders = sorted(
            [folder for folder in main_log_path.iterdir() if folder.is_dir()],
            key=lambda folder: folder.stat().st_mtime,
            reverse=True,
        )
        folders = [folder.name for folder in folders]

        st.selectbox(f"**Select from `{main_log_path.absolute()}`**", folders, key="log_path")

    if st.button("Refresh Data"):
        load_data()
        st.rerun()

    expand_all = st.toggle("Expand All", key="expand_all")


def show_text(text, lang=None):
    if lang is not None:
        st.code(text, language=lang, wrap_lines=True)
    elif "```py" in text:
        st.code(text, language="python", wrap_lines=True)
    else:
        st.code(text, language="html", wrap_lines=True)


def highlight_prompts_uri(uri):
    parts = uri.split(":")
    return f"**{parts[0]}:**:green[**{parts[1]}**]"

def extract_loopid_func_name(tag):
    match = re.search(r'Loop_(\d+)\.(\w+)\.', tag)
    if match:
        return match.group(1), match.group(2)
    return None, None

def extract_evoid(tag):
    match = re.search(r'\.evo_loop_(\d+)\.', tag)
    if match:
        return match.group(1)
    return None

# Display the data
for d in session_state.data:
    tag = d["tag"]
    obj = d["obj"]
    
    loop_id, func_name = extract_loopid_func_name(tag)
    evo_id = extract_evoid(tag)
    if loop_id:
        if loop_id not in tlist:
            tlist.append(loop_id)
            st.subheader(f"**Loop_{loop_id}**", anchor=f"Loop_{loop_id}", divider="blue")
        if f"loop_{loop_id}.{func_name}" not in tlist:
            tlist.append(f"loop_{loop_id}.{func_name}")
            st.subheader(f"**{func_name}**", anchor=f"loop_{loop_id}.{func_name}", divider="green")
        if f"loop_{loop_id}.{evo_id}" not in tlist:
            tlist.append(f"loop_{loop_id}.evo_step_{evo_id}")
            st.subheader(f"**evo_step_{evo_id}**", anchor=f"loop_{loop_id}.evo_step_{evo_id}", divider="orange")

    if "debug_tpl" in tag:
        uri = obj["uri"]
        tpl = obj["template"]
        cxt = obj["context"]
        rd = obj["rendered"]

        with st.expander(highlight_prompts_uri(uri), expanded=expand_all, icon="⚙️"):
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

        with st.expander(f"**LLM**", expanded=expand_all, icon="🤖"):
            t1, t2, t3 = st.tabs([":green[**Response**]", ":blue[**User**]", ":orange[**System**]"])
            with t3:
                if system is None:
                    st.text("In session, no system prompt")
                else:
                    show_text(system)
            with t2:
                show_text(user)
            with t1:
                try:
                    rdict = json.loads(resp)
                    if "code" in rdict:
                        code = rdict["code"]
                        st.markdown(":red[**Code in response dict:**]")
                        st.code(code, language="python", wrap_lines=True)
                        rdict.pop("code")
                    elif "spec" in rdict:
                        spec = rdict["spec"]
                        st.markdown(":red[**Spec in response dict:**]")
                        st.markdown(spec)
                        rdict.pop("spec")
                    st.write(":red[**Other parts (except for the code or spec) in response dict:**]")
                    st.json(rdict)
                except:
                    st.json(resp)

with st.sidebar:
    et_toc = ""
    for t in tlist:
        if t.startswith("L"):
            et_toc += f"- [{t}](#{t})\n"
        elif 'evo_step_' in t:
            et_toc += f"    - [{t}](#{t})\n"
        else:
            et_toc += f"  - [{t}](#{t})\n"
    st.markdown(et_toc, unsafe_allow_html=True)
