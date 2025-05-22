from pathlib import Path

import streamlit as st
from streamlit import session_state as state

from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.log.ui.conf import UI_SETTING


def convert_log_folder_str(lf: str) -> str:
    if "/" not in lf:
        return f"/data/share_folder_local/amlt/{lf.strip()}/combined_logs"
    return lf.strip()


def extract_amlt_name(x: str) -> str:
    if "amlt" not in x:
        return x
    return x[x.rfind("amlt") + 5 :].split("/")[0]


# 设置主日志路径
if "log_folder" not in state:
    state.log_folder = Path("./log")
if "log_folders" not in state:
    state.log_folders = [convert_log_folder_str(i) for i in UI_SETTING.default_log_folders]

summary_page = st.Page("ds_summary.py", title="Summary", icon="📊")
trace_page = st.Page("ds_trace.py", title="Trace", icon="📈")
aide_page = st.Page("aide.py", title="Aide", icon="🧑‍🏫")
st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")
st.navigation([summary_page, trace_page, aide_page]).run()


# UI - Sidebar
with st.sidebar:
    st.subheader("Pages", divider="rainbow")
    st.page_link(summary_page, icon="📊")
    st.page_link(trace_page, icon="📈")
    st.page_link(aide_page, icon="🧑‍🏫")

    st.subheader("Settings", divider="rainbow")
    with st.form("log_folder_form", border=False):
        log_folder_str = st.text_area(
            "**Log Folders**(split by ';')", value=";".join(extract_amlt_name(i) for i in state.log_folders)
        )
        if st.form_submit_button("Confirm"):
            state.log_folders = [
                convert_log_folder_str(folder) for folder in log_folder_str.split(";") if folder.strip()
            ]
            st.rerun()
