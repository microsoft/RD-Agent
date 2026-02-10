"""
RL Post-training Timeline Viewer
Hierarchical view: Session > Loop > Stage > Events

Run:
    streamlit run rdagent/app/rl/ui/app.py
"""

import os
from pathlib import Path

import streamlit as st
from streamlit import session_state as state

from rdagent.app.rl.ui.components import render_session, render_summary
from rdagent.app.rl.ui.config import ALWAYS_VISIBLE_TYPES, OPTIONAL_TYPES
from rdagent.app.rl.ui.data_loader import get_summary, get_valid_sessions, load_session
from rdagent.app.rl.ui.rl_summary import render_job_summary

DEFAULT_LOG_BASE = "log/"


def get_job_options(base_path: Path) -> list[str]:
    """Scan directory and return job options list."""
    options = []
    has_root_tasks = False
    job_dirs = []

    if not base_path.exists():
        return options

    for d in base_path.iterdir():
        if not d.is_dir():
            continue
        if (d / "__session__").exists():
            has_root_tasks = True
        else:
            try:
                if any((sub / "__session__").exists() for sub in d.iterdir() if sub.is_dir()):
                    job_dirs.append(d.name)
            except PermissionError:
                pass

    job_dirs.sort(reverse=True)
    options.extend(job_dirs)
    if has_root_tasks:
        options.append(". (Current)")

    return options


def main():
    st.set_page_config(layout="wide", page_title="RL Timeline", page_icon="🤖")

    with st.sidebar:
        view_mode = st.radio("View Mode", ["Job Summary", "Single Task"], horizontal=True)
        st.divider()

        default_log = os.environ.get("RL_LOG_PATH", DEFAULT_LOG_BASE)
        job_folder = default_log
        selected_types = ALWAYS_VISIBLE_TYPES.copy()
        is_root_job = False

        if view_mode == "Job Summary":
            st.header("Job")
            base_folder = st.text_input("Base Folder", value=default_log, key="base_folder_input")
            base_path = Path(base_folder)

            job_options = get_job_options(base_path)
            if job_options:
                selected_job = st.selectbox("Select Job", job_options, key="job_select")
                if selected_job.startswith("."):
                    job_folder = base_folder
                    is_root_job = True
                else:
                    job_folder = str(base_path / selected_job)
                state.selected_job_folder = job_folder
            else:
                st.warning("No jobs found in this directory")
                job_folder = base_folder

            if st.button("Refresh", type="primary", key="refresh_job"):
                st.rerun()
        else:
            st.header("Session")
            default_path = getattr(state, "selected_job_folder", default_log)
            log_folder = st.text_input("Log Folder", value=default_path)
            log_path = Path(log_folder)

            sessions = get_valid_sessions(log_path)
            if not sessions:
                st.warning("No valid sessions found")
                return

            selected_session = st.selectbox("Session", sessions)

            if st.button("Load", type="primary") or "session" not in state:
                with st.spinner("Loading..."):
                    state.session = load_session(log_path / selected_session)
                    state.session_name = selected_session

            st.divider()

            st.subheader("Show More")
            selected_types = ALWAYS_VISIBLE_TYPES.copy()
            for event_type, (label, default) in OPTIONAL_TYPES.items():
                if st.toggle(label, value=default, key=f"toggle_{event_type}"):
                    selected_types.append(event_type)

            st.divider()

            if "session" in state:
                summary = get_summary(state.session)
                st.subheader("Summary")
                st.metric("Loops", summary.get("loop_count", 0))
                st.metric("LLM Calls", summary.get("llm_call_count", 0))
                success = summary.get("docker_success", 0)
                fail = summary.get("docker_fail", 0)
                st.metric("Docker", f"{success}✓ / {fail}✗")

    if view_mode == "Job Summary":
        st.title("📊 RL Job Summary")
        job_path = Path(job_folder)
        if job_path.exists():
            render_job_summary(job_path, is_root=is_root_job)
        else:
            st.warning(f"Job folder not found: {job_folder}")
        return

    st.title("🤖 RL Timeline Viewer")

    if "session" not in state:
        st.info("Select a session and click **Load** to view")
        return

    session = state.session
    summary = get_summary(session)
    render_summary(summary)
    st.divider()
    render_session(session, selected_types)


if __name__ == "__main__":
    main()

