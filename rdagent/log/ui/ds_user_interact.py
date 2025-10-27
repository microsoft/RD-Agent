import json
import pickle
import random
import time
from datetime import datetime, timedelta
from importlib.resources import files as rfiles
from pathlib import Path

import streamlit as st
from streamlit import session_state as state

from rdagent.app.data_science.conf import DS_RD_SETTING

st.set_page_config(layout="wide", page_title="RD-Agent_user_interact", page_icon="🎓", initial_sidebar_state="expanded")

# 初始化session state
if "sessions" not in state:
    state.sessions = {}
if "selected_session_name" not in state:
    state.selected_session_name = None


def render_main_content():
    """渲染主要内容区域"""
    if state.selected_session_name is not None and state.selected_session_name in state.sessions:
        selected_session_data = state.sessions[state.selected_session_name]
        if selected_session_data is not None:
            show_name = (
                selected_session_data["competition"]
                if "vaccine" not in selected_session_data["competition"]
                else "mRNA-vaccine-degradation"
            )
            st.title(f"Session: {state.selected_session_name[:4]}: {show_name}")
            # st.title("Contextual Information:")
            # st.subheader("Competition scenario:", divider=True)
            st.markdown(
                '#### Target: <span style="color:red; font-weight:700;">to predict the degradation possibility of mRNA vaccine</span>',
                unsafe_allow_html=True,
            )
            scenario = st.image(rfiles("rdagent.log.ui").joinpath("image.png"))
            # scenario = st.code(selected_session_data["scenario_description"], language="yaml")
            st.subheader("Former attempts summary:", divider=True)
            with st.expander("Former hypotheses", expanded=False):
                scenario = st.code(selected_session_data["ds_trace_desc"], language="yaml", wrap_lines=True)
            if selected_session_data["current_code"] != "":
                with st.expander("Current SOTA code", expanded=False):
                    st.subheader("Current SOTA code", divider=True)
                    scenario = st.code(
                        body=selected_session_data["current_code"],
                        language="python",
                    )

            st.subheader("Hypothesis candidates:", divider=True)
            hypothesis_candidates = selected_session_data["hypothesis_candidates"]
            tabs_names = [
                f"{'✅' if i == selected_session_data['target_hypothesis_index'] or selected_session_data['target_hypothesis_index'] == -1 else ''}Hypothesis {i+1}"
                for i in range(len(hypothesis_candidates))
            ]
            target_tab_names = [name for name in tabs_names if name.startswith("✅")][0]
            tabs = st.tabs(tabs_names, default=target_tab_names)
            for index, hypothesis in enumerate(hypothesis_candidates):
                with tabs[index]:
                    st.code(str(hypothesis), language="yaml", wrap_lines=True)
            st.text("✅ means picked as target hypothesis")

            st.subheader("Decisions to make:")

            with st.form(key="user_form"):
                st.caption("Please modify the fields below and submit to provide your feedback.")
                with st.expander("Modify target hypothesis and task description", expanded=False):
                    target_hypothesis = st.text_area(
                        "Target hypothesis: (you can copy from candidates)",
                        value=(original_hypothesis := selected_session_data["target_hypothesis"].hypothesis),
                        height="content",
                    )
                    target_task = st.text_area(
                        "Target task description:",
                        value=(original_task_desc := selected_session_data["task"].description),
                        height="content",
                    )
                original_user_instruction = selected_session_data.get("user_instruction")
                user_instruction_list = []
                if selected_session_data.get("former_user_instructions") is not None:
                    st.caption(
                        "Former user instructions, you can modify or delete the content to remove certain instruction."
                    )
                    for user_instruction in selected_session_data.get("former_user_instructions"):
                        user_instruction_list.append(
                            st.text_area("Former user instruction", value=user_instruction, height="content")
                        )
                instruction_text_area = st.text_area("Add new user instruction", value="", height="content")
                user_instruction_list.append(instruction_text_area)
                if "vaccine" in selected_session_data["competition"]:
                    # examples = get_suggestion_with_session_id(state.selected_session_name)
                    examples = [
                        "Use XGBoost for stacking multiple models to improve final predictions.",
                        "Try different model type: Transformer-based models for sequence data.",
                        "Try LSTM.",
                        "Try GRU.",
                        "Use 3D RNA structures to compute more accurate distance matrices.",
                        "Add LSTM or GRU layers after GNN to improve model performance.",
                        "Reverse input sequences during training and testing for data augmentation.",
                        "Apply stratified 5-fold CV based on sequence edit distance.",
                    ]
                    st.caption("Some suggestions for new user instructions:")
                    select_radio = st.radio("Suggestion examples:", examples, index=None)
                    alias_input = st.text_input("Please input your alias here:")
                submit = st.form_submit_button("Submit")
                approve = st.form_submit_button("Approve without changes")

                if submit or approve:
                    if approve:
                        submit_dict = {
                            "action": "confirm",
                        }
                    else:
                        user_instruction_str_list = [ui for ui in user_instruction_list if ui.strip() != ""]
                        user_instruction_str_list = (
                            None if len(user_instruction_str_list) == 0 else user_instruction_str_list
                        )
                        action = (
                            "confirm"
                            if target_hypothesis == original_hypothesis
                            and target_task == original_task_desc
                            and user_instruction_str_list == original_user_instruction
                            else "rewrite"
                        )
                        submit_dict = {
                            "target_hypothesis": target_hypothesis,
                            "task_description": target_task,
                            "user_instruction": user_instruction_str_list,
                            "action": action,
                            "chosen_example": (
                                select_radio if "vaccine" in selected_session_data["competition"] else None
                            ),
                            "alias": (alias_input if "vaccine" in selected_session_data["competition"] else None),
                        }
                    json.dump(
                        submit_dict,
                        open(
                            DS_RD_SETTING.user_interaction_mid_folder / f"{state.selected_session_name}_RET.json", "w"
                        ),
                    )
                    json.dump(
                        submit_dict,
                        open(
                            DS_RD_SETTING.user_interaction_mid_folder
                            / f"../RD-Agent_user_interaction_tab/{state.selected_session_name}_RET.json",
                            "w",
                        ),
                    )
                    Path(DS_RD_SETTING.user_interaction_mid_folder / f"{state.selected_session_name}.pkl").unlink(
                        missing_ok=True
                    )
                    Path(
                        DS_RD_SETTING.user_interaction_mid_folder / f"{state.selected_session_name}_suggestions.json"
                    ).unlink(missing_ok=True)
                    st.success("Your feedback has been submitted. Thank you!")
                    time.sleep(5)
                    state.selected_session_name = None

            if st.button("Extend expiration by 60s"):
                session_data = pickle.load(
                    open(DS_RD_SETTING.user_interaction_mid_folder / f"{state.selected_session_name}.pkl", "rb")
                )
                session_data["expired_datetime"] = session_data["expired_datetime"] + timedelta(seconds=60)
                pickle.dump(
                    session_data,
                    open(DS_RD_SETTING.user_interaction_mid_folder / f"{state.selected_session_name}.pkl", "wb"),
                )
    else:
        st.warning("Please select a session from the sidebar.")


# 每秒更新一次sessions
@st.fragment(run_every=1)
def update_sessions():
    log_folder = Path(DS_RD_SETTING.user_interaction_mid_folder)
    state.sessions = {}
    for session_file in log_folder.glob("*.pkl"):
        try:
            session_data = pickle.load(open(session_file, "rb"))
            if session_data["expired_datetime"] > datetime.now():
                state.sessions[session_file.stem] = session_data
            else:
                session_file.unlink(missing_ok=True)
                ret_file = log_folder / f"{session_file.stem}_RET.json"
                ret_file.unlink(missing_ok=True)
        except Exception as e:
            continue
    render_main_content()


@st.fragment(run_every=1)
def render_sidebar():
    # Manage sidebar collapse state
    if "sidebar_collapsed" not in state:
        state.sidebar_collapsed = False
    # If no session selected anymore (e.g., after submit), re-expand
    if state.selected_session_name is None and state.sidebar_collapsed:
        state.sidebar_collapsed = False

    # If collapsed, inject CSS to hide sidebar area (cannot change set_page_config dynamically)
    if state.sidebar_collapsed:
        st.markdown(
            """
            <style>
            section[data-testid="stSidebar"] {display: none !important;}
            div[data-testid="collapsedControl"] {display: none !important;}
            </style>
            """,
            unsafe_allow_html=True,
        )
        return  # Do not render sidebar content while collapsed

    st.title("R&D-Agent User Interaction Portal")
    if state.sessions:
        st.header("Active Sessions")
        st.caption("Click a session to view:")
        session_names = [name for name in state.sessions]
        for session_name in session_names:
            with st.container(border=True):
                remaining = state.sessions[session_name]["expired_datetime"] - datetime.now()
                total_sec = int(remaining.total_seconds())
                label = f"{total_sec}s to expire" if total_sec > 0 else "Expired"
                if st.button(f"session id:{session_name[:4]}", key=f"session_btn_{session_name}"):
                    state.selected_session_name = session_name
                    state.data = state.sessions[session_name]
                    state.sidebar_collapsed = True  # Collapse after click
                st.markdown(f"⏳ {label}")
    else:
        st.warning("No active sessions available. Please wait.")
    if st.button("Clear all sessions"):
        for file_path in Path(DS_RD_SETTING.user_interaction_mid_folder).iterdir():
            file_path.unlink(missing_ok=True)


update_sessions()
with st.sidebar:
    render_sidebar()
