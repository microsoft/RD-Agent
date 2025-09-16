import json
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

from rdagent.app.data_science.conf import DS_RD_SETTING

st.set_page_config(layout="wide", page_title="RD-Agent_user_interact", page_icon="🎓", initial_sidebar_state="expanded")


def get_all_interaction_sessions():
    log_folder = Path(DS_RD_SETTING.user_interaction_mid_folder)
    sessions = {}
    for session_file in log_folder.glob("*.pkl"):
        try:
            session_data = pickle.load(open(session_file, "rb"))
            if session_data["expired_datetime"] > datetime.now():
                sessions[session_file.stem] = session_data
        except Exception as e:
            continue

    return sessions


if "selected_session_name" not in st.session_state:
    st.session_state.selected_session_name = None
with st.sidebar:
    st.title("R&D-Agent User Interaction Portal")
    st.session_state.sessions = get_all_interaction_sessions()
    print(st.session_state.sessions)
    if st.session_state.sessions:
        st.header("Active Sessions")
        st.caption("Click a session to view:")
        session_names = [name for name in st.session_state.sessions]
        for session_name in session_names:
            with st.container(border=True):
                remaining = st.session_state.sessions[session_name]["expired_datetime"] - datetime.now()
                total_sec = int(remaining.total_seconds())
                label = f"{total_sec}s to expire" if total_sec > 0 else "Expired"
                if st.button(f"session id:{session_name[:4]}", key=f"session_btn_{session_name}"):
                    st.session_state.selected_session_name = session_name
                    st.session_state.data = st.session_state.sessions[session_name]
                st.markdown(f"⏳ {label}")

        selected_session_name = st.session_state.selected_session_name
        st.session_state.refresh = False
    else:
        selected_session_name = None
        st.warning("No active sessions available. Please wait.")
        st.session_state.refresh = True

if selected_session_name is not None:
    selected_session_data = st.session_state.sessions[selected_session_name]
    if selected_session_data is not None:
        st.title(f"Session: {selected_session_name[:4]} with competition {selected_session_data['competition']}")
        st.title("Contextual Information:")
        st.subheader("Competition scenario:", divider=True)
        scenario = st.code(selected_session_data["scenario_description"], language="yaml")
        st.subheader("Former attempts summary:", divider=True)
        scenario = st.code(selected_session_data["ds_trace_desc"], language="yaml")
        if selected_session_data["current_code"] != "":
            st.subheader("Current SOTA code", divider=True)
            scenario = st.code(
                body=selected_session_data["current_code"],
                language="python",
            )

        st.subheader("Hypothesis candidates:", divider=True)
        hypothesis_candidates = selected_session_data["hypothesis_candidates"]
        tabs = st.tabs(
            [
                f"{'✅' if i == selected_session_data['target_hypothesis_index'] or selected_session_data['target_hypothesis_index'] == -1 else ''}Hypothesis {i+1}"
                for i in range(len(hypothesis_candidates))
            ]
        )
        for index, hypothesis in enumerate(hypothesis_candidates):
            with tabs[index]:
                st.code(str(hypothesis), language="yaml")
        st.text("✅ means picked as target hypothesis")

        st.title("Decisions to make:")

        with st.form(key="user_form"):
            st.caption("Please modify the fields below and submit to provide your feedback.")
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
            user_instruction = st.text_area(
                "Additional instructions for the agent: (Highest priority)", value="", height="content"
            )
            submit = st.form_submit_button("Submit")

            if submit:
                action = (
                    "confirm"
                    if target_hypothesis == original_hypothesis
                    and target_task == original_task_desc
                    and user_instruction == ""
                    else "rewrite"
                )
                submit_dict = {
                    "target_hypothesis": target_hypothesis,
                    "task_description": target_task,
                    "user_instruction": user_instruction,
                    "action": action,
                }
                json.dump(
                    submit_dict,
                    open(DS_RD_SETTING.user_interaction_mid_folder / f"{selected_session_name}_RET.json", "w"),
                )
                Path(DS_RD_SETTING.user_interaction_mid_folder / f"{selected_session_name}.pkl").unlink(missing_ok=True)
                st.success("Your feedback has been submitted. Thank you!")
                time.sleep(5)
                st.session_state.selected_session_name = None
                st.session_state.refresh = True

        if st.button("Extend expiration by 60s"):
            session_data = pickle.load(
                open(DS_RD_SETTING.user_interaction_mid_folder / f"{selected_session_name}.pkl", "rb")
            )
            session_data["expired_datetime"] = session_data["expired_datetime"] + timedelta(seconds=60)
            pickle.dump(
                session_data,
                open(DS_RD_SETTING.user_interaction_mid_folder / f"{selected_session_name}.pkl", "wb"),
            )
else:
    st.warning("Please select a session from the sidebar.")
    st.session_state.refresh = True

if st.session_state.refresh:
    time.sleep(1)
    st.rerun()
