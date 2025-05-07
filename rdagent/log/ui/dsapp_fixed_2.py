from pathlib import Path

import streamlit as st
from streamlit import session_state as state

from rdagent.app.data_science.loop import DataScienceRDLoop

# 设置主日志路径
if "log_folder" not in state:
    state.log_folder = Path("./log")
if "log_folders" not in state:
    state.log_folders = ["/home/xuyang1/workspace/JobAndExp/amlt_project/amlt/apparent-zebra/combined_logs"]

summary_page = st.Page("ds_summary_fixed_2.py", title="Summary", icon="📊")
trace_page = st.Page("ds_trace_fixed_2.py", title="Trace", icon="📈")
st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")
st.navigation([summary_page, trace_page]).run()

# UI - Sidebar
with st.sidebar:
    st.subheader("Pages", divider="rainbow")
    st.page_link(summary_page, icon="📊")
    st.page_link(trace_page, icon="📈")
