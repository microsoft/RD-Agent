"""
FT (Fine-tune) 场景前端可视化
用于分析 LLM 微调实验的历史记录

运行命令：
    streamlit run rdagent/app/finetune/llm/ui/app.py
"""

from pathlib import Path

import streamlit as st
from streamlit import session_state as state

from rdagent.app.finetune.llm.ui.components import (
    show_evo_loops,
    show_feedback,
    show_files,
    show_hypothesis,
    show_results,
    show_runner_result,
    show_scenario_info,
)
from rdagent.app.finetune.llm.ui.data_loader import get_valid_sessions, load_ft_data


def main():
    st.set_page_config(layout="wide", page_title="FT Trace Viewer", page_icon="FT")

    # ========== Sidebar ==========
    with st.sidebar:
        st.header("Settings")

        log_folder = st.text_input("Log Folder", value="./log")
        log_path = Path(log_folder)

        sessions = get_valid_sessions(log_path)
        if not sessions:
            st.warning("No valid sessions found")
            return

        selected_session = st.selectbox("Session", sessions)

        if st.button("Load Data") or "ft_data" not in state:
            with st.spinner("Loading..."):
                state.ft_data = load_ft_data(log_path / selected_session)
                state.session_name = selected_session

        st.divider()

        # 显示加载状态
        if "ft_data" in state:
            loops = state.ft_data.get("loops", {})
            st.metric("Loops", len(loops))

    # ========== Main Content ==========
    st.title("FT (Fine-tune) Trace Viewer")

    if "ft_data" not in state:
        st.info("Please select a session and click 'Load Data'")
        return

    data = state.ft_data
    loops = data.get("loops", {})

    if not loops:
        st.warning("No loop data found in this session")
        show_scenario_info(data.get("scenario"))
        return

    # 场景信息
    show_scenario_info(data.get("scenario"))

    st.divider()

    # Loop 选择
    loop_ids = sorted(loops.keys())
    if len(loop_ids) > 1:
        loop_id = st.select_slider("Loop", options=loop_ids, value=loop_ids[0])
    else:
        loop_id = loop_ids[0]
        st.info(f"Loop {loop_id}")

    loop_data = loops[loop_id]
    evo_loops = loop_data.get("evo_loops", {})

    # ========== Hypothesis ==========
    show_hypothesis(loop_data.get("experiment"))

    st.divider()

    # ========== Generated Files（纵向排布）==========
    if evo_loops:
        last_evo_id = max(evo_loops.keys())
        last_code = evo_loops[last_evo_id].get("code")
        show_files(last_code)
    else:
        show_files(None)

    st.divider()

    # ========== Results（纵向排布）==========
    if evo_loops:
        last_evo_id = max(evo_loops.keys())
        last_code = evo_loops[last_evo_id].get("code")
        show_results(last_code)
    else:
        show_results(None)

    st.divider()

    # ========== Evolution Loops ==========
    show_evo_loops(evo_loops)

    st.divider()

    # ========== Full Train Result ==========
    show_runner_result(loop_data.get("runner_result"))

    st.divider()

    # ========== Final Feedback ==========
    show_feedback(loop_data.get("feedback"))


if __name__ == "__main__":
    main()
