"""
FT (Fine-tune) Timeline Viewer
Hierarchical view: Session > Loop > Stage > EvoLoop > Events

Run:
    streamlit run rdagent/app/finetune/llm/ui/app.py
"""

from pathlib import Path

import streamlit as st
from streamlit import session_state as state

from rdagent.app.finetune.llm.ui.components import render_session, render_summary
from rdagent.app.finetune.llm.ui.data_loader import get_summary, get_valid_sessions, load_ft_session

# Always visible types
ALWAYS_VISIBLE = ["scenario", "dataset_selection", "llm_call", "experiment", "code", "docker_exec", "feedback"]

# Optional types with toggles (label, default)
OPTIONAL_TYPES = {
    "template": ("📋 Template", False),
    "token": ("🔢 Token", False),
    "time": ("⏱️ Time", False),
    "settings": ("⚙️ Settings", False),
}


def main():
    st.set_page_config(layout="wide", page_title="FT Timeline", page_icon="🔬")

    # Enable word wrap for code blocks (like VSCode)
    st.markdown("""
    <style>
    /* Target all code elements in Streamlit */
    .stCodeBlock pre,
    .stCodeBlock code,
    .stCodeBlock [data-testid="stCodeBlock"] pre,
    .stCodeBlock [data-testid="stCodeBlock"] code,
    code[class*="language-"],
    pre[class*="language-"],
    .hljs {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
    }
    /* Ensure the container doesn't force horizontal scroll */
    .stCodeBlock,
    .stCodeBlock > div,
    [data-testid="stCodeBlock"],
    [data-testid="stCodeBlock"] > div {
        overflow-x: visible !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ========== Sidebar ==========
    with st.sidebar:
        st.header("Session")

        log_folder = st.text_input("Log Folder", value="./log")
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
        selected_types = ALWAYS_VISIBLE.copy()
        for event_type, (label, default) in OPTIONAL_TYPES.items():
            if st.toggle(label, value=default, key=f"toggle_{event_type}"):
                selected_types.append(event_type)

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

    # Summary bar
    render_summary(summary)

    st.divider()

    # Hierarchical view
    render_session(session, selected_types)


if __name__ == "__main__":
    main()
