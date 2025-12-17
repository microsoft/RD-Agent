"""
FT (Fine-tune) Timeline Viewer
Hierarchical view: Session > Loop > Stage > EvoLoop > Events

Run:
    streamlit run rdagent/app/finetune/llm/ui/app.py
"""

import os
from pathlib import Path

import streamlit as st
from streamlit import session_state as state

from rdagent.app.finetune.llm.ui.components import render_session, render_summary
from rdagent.app.finetune.llm.ui.config import ALWAYS_VISIBLE_TYPES, OPTIONAL_TYPES
from rdagent.app.finetune.llm.ui.data_loader import (
    get_summary,
    get_valid_sessions,
    load_ft_session,
)


def main():
    st.set_page_config(layout="wide", page_title="FT Timeline", page_icon="🔬")

    # ========== Sidebar ==========
    with st.sidebar:
        st.header("Session")

        default_log = os.environ.get("FT_LOG_PATH", "./log")
        log_folder = st.text_input("Log Folder", value=default_log)
        log_path = Path(log_folder)

        sessions = get_valid_sessions(log_path)
        if not sessions:
            st.warning("No valid sessions found")
            return

        selected_session = st.selectbox("Session", sessions)

        if st.button("Load", type="primary") or "session" not in state:
            with st.spinner("Loading..."):
                state.session = load_ft_session(log_path / selected_session)
                state.session_name = selected_session

        st.divider()

        # Optional type toggles
        st.subheader("Show More")
        selected_types = ALWAYS_VISIBLE_TYPES.copy()
        for event_type, (label, default) in OPTIONAL_TYPES.items():
            if st.toggle(label, value=default, key=f"toggle_{event_type}"):
                selected_types.append(event_type)

        st.divider()

        # Display options
        st.subheader("Display Options")
        state.render_markdown = st.toggle("Render Prompts", value=False, key="render_markdown_toggle")

        st.divider()

        # Summary in sidebar
        if "session" in state:
            summary = get_summary(state.session)
            st.subheader("Summary")
            st.metric("Loops", summary.get("loop_count", 0))
            st.metric("LLM Calls", summary.get("llm_call_count", 0))
            success = summary.get("docker_success", 0)
            fail = summary.get("docker_fail", 0)
            st.metric("Docker", f"{success}✓ / {fail}✗")

    # ========== Main Content ==========
    st.title("🔬 FT Timeline Viewer")

    if "session" not in state:
        st.info("Select a session and click **Load** to view")
        return

    session = state.session
    summary = get_summary(session)

    # Global info header (Base Model, Datasets, Benchmark) - compact style
    scenario_event = next((e for e in session.init_events if e.type == "scenario"), None)
    dataset_event = next((e for e in session.init_events if e.type == "dataset_selection"), None)

    if scenario_event or dataset_event:
        if scenario_event and hasattr(scenario_event.content, "base_model"):
            st.markdown(f"🧠 **Model:** `{scenario_event.content.base_model}`")
        if dataset_event:
            selected = dataset_event.content.get("selected_datasets", []) if isinstance(dataset_event.content, dict) else []
            if selected:
                st.markdown(f"📂 **Datasets:** `{', '.join(selected)}`")
        if scenario_event and hasattr(scenario_event.content, "target_benchmark"):
            st.markdown(f"🎯 **Benchmark:** `{scenario_event.content.target_benchmark}`")

    # Summary bar
    render_summary(summary)

    st.divider()

    # Hierarchical view
    render_session(session, selected_types)


if __name__ == "__main__":
    main()
