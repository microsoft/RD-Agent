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

eset = set()


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


# Display the data
for d in session_state.data:
    tag = d["tag"]
    obj = d["obj"]
    if "evo_loop_" in tag:
        tags = tag.split(".")
        for t in tags:
            if "evo_loop_" in t:
                etag = t
                break
        if etag not in eset:
            eset.add(etag)
            st.subheader(f"**{etag}**", anchor=etag, divider="rainbow")
    if "debug_tpl" in tag:
        uri = obj["uri"]
        tpl = obj["template"]
        cxt = obj["context"]
        rd = obj["rendered"]

        with st.expander(highlight_prompts_uri(uri), expanded=expand_all, icon="⚙️"):
            t1, t2, t3 = st.tabs([":blue[**Template**]", ":orange[**Context**]", ":green[**Rendered**]"])
            with t1:
                show_text(tpl, lang="django")
            with t2:
                st.json(cxt)
            with t3:
                show_text(rd)
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
    et_toc = "\n".join(f"- [**{etag}**](#{etag})" for etag in sorted(eset))
    st.markdown(et_toc, unsafe_allow_html=True)
