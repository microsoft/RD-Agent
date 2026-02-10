"""
RL UI Components - Event Renderers
Simplified version without EvoLoop
"""

from typing import Any

import streamlit as st

from rdagent.app.rl.ui.config import ICONS
from rdagent.app.rl.ui.data_loader import Event, Loop, Session


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def render_session(session: Session, show_types: list[str]) -> None:
    """Render full session"""
    if session.init_events:
        filtered = [e for e in session.init_events if e.type in show_types]
        if filtered:
            with st.expander("🚀 **Initialization**", expanded=False):
                for event in filtered:
                    render_event(event)

    for loop_id in sorted(session.loops.keys()):
        loop = session.loops[loop_id]
        render_loop(loop, show_types)


def render_loop(loop: Loop, show_types: list[str]) -> None:
    """Render a single loop"""
    # Get status indicators
    docker_success = None
    feedback_decision = None
    
    for event in loop.running:
        if event.type == "docker_exec" and event.success is not None:
            docker_success = event.success
    
    for event in loop.feedback:
        if event.type == "feedback" and event.success is not None:
            feedback_decision = event.success

    # Build title
    parts = []
    if docker_success is not None:
        parts.append("🐳✓" if docker_success else "🐳✗")
    if feedback_decision is not None:
        parts.append("✅" if feedback_decision else "❌")
    
    result_str = " ".join(parts) if parts else ""

    with st.expander(f"🔄 **Loop {loop.loop_id}** {result_str}", expanded=False):
        # Proposal
        if loop.proposal:
            filtered = [e for e in loop.proposal if e.type in show_types]
            if filtered:
                st.markdown("#### 💡 Proposal")
                for event in filtered:
                    render_event(event)

        # Coding
        if loop.coding:
            filtered = [e for e in loop.coding if e.type in show_types]
            if filtered:
                st.markdown("#### 💻 Coding")
                for event in filtered:
                    render_event(event)

        # Running
        if loop.running:
            filtered = [e for e in loop.running if e.type in show_types]
            if filtered:
                st.markdown("#### 🏃 Running")
                for event in filtered:
                    render_event(event)

        # Feedback
        if loop.feedback:
            filtered = [e for e in loop.feedback if e.type in show_types]
            if filtered:
                st.markdown("#### 📊 Feedback")
                for event in filtered:
                    render_event(event)


def render_event(event: Event) -> None:
    """Render a single event"""
    icon = ICONS.get(event.type, "📌")
    duration_str = f" ({format_duration(event.duration)})" if event.duration else ""

    status = ""
    if event.success is True:
        status = "🟢 "
    elif event.success is False:
        status = "🔴 "

    title = f"{event.time_str} {icon} {status}{event.title}{duration_str}"

    renderers = {
        "scenario": render_scenario,
        "llm_call": render_llm_call,
        "template": render_template,
        "experiment": render_experiment,
        "code": render_code,
        "docker_exec": render_docker_exec,
        "feedback": render_feedback,
        "token": render_token,
        "time": render_time_info,
        "settings": render_settings,
        "hypothesis": render_hypothesis,
    }

    renderer = renderers.get(event.type, render_generic)
    with st.expander(title, expanded=False):
        renderer(event.content)


def render_scenario(content: Any) -> None:
    if hasattr(content, "base_model"):
        st.markdown(f"**Base Model:** `{content.base_model}`")
    if hasattr(content, "benchmark"):
        st.markdown(f"**Benchmark:** `{content.benchmark}`")
    render_generic(content)


def render_hypothesis(content: Any) -> None:
    if hasattr(content, "hypothesis") and content.hypothesis:
        st.markdown("**Hypothesis:**")
        st.markdown(content.hypothesis)
    if hasattr(content, "reason") and content.reason:
        with st.expander("Reason", expanded=False):
            st.markdown(content.reason)


def render_settings(content: Any) -> None:
    if isinstance(content, dict):
        st.json(content)
    else:
        st.code(str(content), wrap_lines=True)


def render_llm_call(content: Any) -> None:
    if not isinstance(content, dict):
        st.json(content) if content else st.info("No content")
        return

    if content.get("start") and content.get("end"):
        duration = (content["end"] - content["start"]).total_seconds()
        st.caption(f"Duration: {format_duration(duration)}")

    system = content.get("system", "")
    if system:
        with st.expander("System Prompt", expanded=False):
            st.code(system, language="text", line_numbers=True, wrap_lines=True)

    user = content.get("user", "")
    if user:
        with st.expander("User Prompt", expanded=False):
            st.code(user, language="text", line_numbers=True, wrap_lines=True)

    resp = content.get("resp", "")
    if resp:
        st.markdown("**Response:**")
        if resp.strip().startswith("{") or resp.strip().startswith("["):
            st.code(resp, language="json", line_numbers=True, wrap_lines=True)
        elif resp.strip().startswith("```"):
            st.markdown(resp)
        else:
            st.code(resp, language="text", line_numbers=True, wrap_lines=True)


def render_template(content: Any) -> None:
    if not isinstance(content, dict):
        st.json(content) if content else st.info("No content")
        return

    uri = content.get("uri", "")
    st.caption(f"URI: `{uri}`")

    context = content.get("context", {})
    if context:
        with st.expander("Context Variables", expanded=False):
            st.json(context)

    rendered = content.get("rendered", "")
    if rendered:
        with st.expander("Rendered", expanded=True):
            st.code(rendered, language="text", line_numbers=True, wrap_lines=True)


def render_experiment(content: Any) -> None:
    if isinstance(content, list):
        for i, task in enumerate(content):
            if len(content) > 1:
                st.markdown(f"**Task {i}**")
            if hasattr(task, "description") and task.description:
                st.markdown(task.description)
    else:
        render_generic(content)


def render_code(content: Any) -> None:
    if isinstance(content, list):
        for ws in content:
            if hasattr(ws, "file_dict") and ws.file_dict:
                for filename, code in ws.file_dict.items():
                    lang = "yaml" if filename.endswith((".yaml", ".yml")) else "python"
                    with st.expander(filename, expanded=False):
                        st.code(code, language=lang, line_numbers=True, wrap_lines=True)
    elif hasattr(content, "file_dict") and content.file_dict:
        for filename, code in content.file_dict.items():
            lang = "yaml" if filename.endswith((".yaml", ".yml")) else "python"
            with st.expander(filename, expanded=False):
                st.code(code, language=lang, line_numbers=True, wrap_lines=True)
    else:
        render_generic(content)


def render_docker_exec(content: Any) -> None:
    if isinstance(content, dict):
        exit_code = content.get("exit_code")
        if exit_code is not None:
            if exit_code == 0:
                st.success(f"Exit code: {exit_code}")
            else:
                st.error(f"Exit code: {exit_code}")

        stdout = content.get("stdout", "")
        if stdout:
            with st.expander("Output", expanded=True):
                st.code(stdout, language="text", line_numbers=True, wrap_lines=True)
    else:
        render_generic(content)


def render_feedback(content: Any) -> None:
    # Handle benchmark result (dict)
    if isinstance(content, dict):
        if "accuracy" in content or "accuracy_summary" in content:
            st.markdown("**Benchmark Result:**")
            st.json(content)
        else:
            st.json(content)
        return

    # Handle HypothesisFeedback object
    col1, col2 = st.columns(2)
    with col1:
        decision = getattr(content, "decision", None)
        if decision is not None:
            st.metric("Decision", "Accept" if decision else "Reject")

    reason = getattr(content, "reason", None)
    if reason:
        with st.expander("Reason", expanded=True):
            st.code(reason, language="text", line_numbers=True, wrap_lines=True)

    code_change = getattr(content, "code_change_summary", None)
    if code_change:
        with st.expander("Code Change Summary", expanded=False):
            st.markdown(code_change)


def render_token(content: Any) -> None:
    if isinstance(content, dict):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prompt", content.get("prompt_tokens", 0))
        with col2:
            st.metric("Completion", content.get("completion_tokens", 0))
        with col3:
            st.metric("Total", content.get("total_tokens", 0))
    else:
        render_generic(content)


def render_time_info(content: Any) -> None:
    if isinstance(content, dict):
        for k, v in content.items():
            st.metric(k, f"{v:.1f}s" if isinstance(v, (int, float)) else str(v))
    else:
        render_generic(content)


def render_generic(content: Any) -> None:
    if hasattr(content, "__dict__"):
        st.json(vars(content))
    elif content:
        st.json(content)
    else:
        st.info("No content")


def render_summary(summary: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Loops", summary.get("loop_count", 0))
    with col2:
        st.metric("LLM Calls", summary.get("llm_call_count", 0))
    with col3:
        llm_time = summary.get("llm_total_time", 0)
        st.metric("LLM Time", format_duration(llm_time))
    with col4:
        success = summary.get("docker_success", 0)
        fail = summary.get("docker_fail", 0)
        st.metric("Docker", f"{success}✓ / {fail}✗")

