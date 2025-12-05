"""
FT UI Components - Hierarchical Event Renderers
"""

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from rdagent.app.finetune.llm.ui.data_loader import Event, EvoLoop, Loop, Session

# Event type icons
ICONS = {
    "scenario": "🎯",
    "llm_call": "💬",
    "template": "📋",
    "experiment": "🧪",
    "code": "📄",
    "docker_exec": "🐳",
    "feedback": "📊",
    "token": "🔢",
    "time": "⏱️",
    "settings": "⚙️",
    "hypothesis": "💡",
    "dataset_selection": "📂",
}


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def render_session(session: Session, show_types: list[str]) -> None:
    """Render full session with hierarchy"""
    # Init events (before any loop)
    if session.init_events:
        filtered = [e for e in session.init_events if e.type in show_types]
        if filtered:
            with st.expander("🚀 **Initialization**", expanded=False):
                for event in filtered:
                    render_event(event)

    # Loops
    for loop_id in sorted(session.loops.keys()):
        loop = session.loops[loop_id]
        render_loop(loop, show_types)


def render_loop(loop: Loop, show_types: list[str]) -> None:
    """Render a single loop with its stages"""
    # Count successes/failures for loop header
    evo_results = []
    for evo in loop.coding.values():
        if evo.success is True:
            evo_results.append("✓")
        elif evo.success is False:
            evo_results.append("✗")
    result_str = " ".join(evo_results) if evo_results else ""

    with st.expander(f"🔄 **Loop {loop.loop_id}** {result_str}", expanded=True):
        # Exp Gen
        if loop.exp_gen:
            filtered = [e for e in loop.exp_gen if e.type in show_types]
            if filtered:
                st.markdown("#### 🧪 Experiment Generation")
                for event in filtered:
                    render_event(event)

        # Coding (Evo Loops)
        if loop.coding:
            st.markdown("#### 💻 Coding")
            for evo_id in sorted(loop.coding.keys()):
                evo = loop.coding[evo_id]
                render_evo_loop(evo, show_types)

        # Runner
        if loop.runner:
            filtered = [e for e in loop.runner if e.type in show_types]
            if filtered:
                st.markdown("#### 🏃 Running(Full Train)")
                for event in filtered:
                    render_event(event)

        # Feedback
        if loop.feedback:
            filtered = [e for e in loop.feedback if e.type in show_types]
            if filtered:
                st.markdown("#### 📊 Feedback")
                for event in filtered:
                    render_event(event)


def render_evo_loop(evo: EvoLoop, show_types: list[str]) -> None:
    """Render evolution loop"""
    filtered = [e for e in evo.events if e.type in show_types]
    if not filtered:
        return

    status = "🟢" if evo.success else "🔴" if evo.success is False else "⚪"
    with st.expander(f"{status} Evo {evo.evo_id}", expanded=False):
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
        "dataset_selection": render_dataset_selection,
    }

    renderer = renderers.get(event.type, render_generic)
    with st.expander(title, expanded=False):
        renderer(event.content)


def render_scenario(content: Any) -> None:
    col1, col2 = st.columns(2)
    with col1:
        if hasattr(content, "base_model"):
            st.metric("Base Model", content.base_model)
        if hasattr(content, "dataset"):
            st.metric("Dataset", content.dataset)
    with col2:
        if hasattr(content, "target_benchmark"):
            st.metric("Target Benchmark", content.target_benchmark)
        if hasattr(content, "device_info"):
            st.text(f"Device: {content.device_info}")


def render_dataset_selection(content: Any) -> None:
    if not isinstance(content, dict):
        st.json(content) if content else st.info("No content")
        return

    selected = content.get("selected_datasets", [])
    total = content.get("total_datasets", 0)
    reasoning = content.get("reasoning", "")

    st.metric("Selected", f"{len(selected)} / {total}")

    if selected:
        st.markdown("**Selected Datasets:**")
        for ds in selected:
            st.markdown(f"- `{ds}`")

    if reasoning:
        with st.expander("Selection Reasoning", expanded=True):
            st.markdown(reasoning)


def render_hypothesis(content: Any) -> None:
    if hasattr(content, "base_model"):
        st.metric("Base Model", content.base_model)
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
        st.code(str(content))


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
            st.code(system, language="text", line_numbers=True)

    user = content.get("user", "")
    if user:
        with st.expander("User Prompt", expanded=True):
            st.code(user, language="text", line_numbers=True)

    resp = content.get("resp", "")
    if resp:
        st.markdown("**Response:**")
        if resp.strip().startswith("{") or resp.strip().startswith("["):
            st.code(resp, language="json", line_numbers=True)
        elif resp.strip().startswith("```"):
            st.markdown(resp)
        else:
            st.code(resp, language="text", line_numbers=True)


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

    template = content.get("template", "")
    if template:
        with st.expander("Template", expanded=False):
            st.code(template, language="text", line_numbers=True)

    rendered = content.get("rendered", "")
    if rendered:
        with st.expander("Rendered", expanded=True):
            st.code(rendered, language="text", line_numbers=True)


def render_experiment(content: Any) -> None:
    if isinstance(content, list):
        for i, task in enumerate(content):
            if len(content) > 1:
                st.markdown(f"**Task {i}**")

            if hasattr(task, "base_model"):
                st.metric("Base Model", task.base_model)

            if hasattr(task, "involving_datasets") and task.involving_datasets:
                st.markdown("**Datasets:**")
                for ds in task.involving_datasets:
                    st.markdown(f"- `{ds}`")

            if hasattr(task, "description") and task.description:
                st.markdown("**Description:**")
                st.markdown(task.description)
    else:
        st.json(content) if content else st.info("No content")


def render_code(content: Any) -> None:
    if not isinstance(content, list):
        st.info("No code available")
        return

    for i, ws in enumerate(content):
        if not hasattr(ws, "file_dict") or not ws.file_dict:
            continue

        if len(content) > 1:
            st.markdown(f"**Workspace {i}**")

        for filename, code in ws.file_dict.items():
            lang = "yaml" if filename.endswith((".yaml", ".yml")) else "python"
            with st.expander(filename, expanded=False):
                st.code(code, language=lang, line_numbers=True)


def render_docker_exec(content: Any) -> None:
    # CoSTEERMultiFeedback (evolving feedback)
    if hasattr(content, "feedback_list"):
        for i, fb in enumerate(content.feedback_list):
            if len(content.feedback_list) > 1:
                st.markdown(f"**Feedback {i}**")

            decision = getattr(fb, "final_decision", None)
            if decision is True:
                st.success("Execution: PASS")
            elif decision is False:
                st.error("Execution: FAIL")

            execution = getattr(fb, "execution", "")
            if execution:
                with st.expander("Execution Log", expanded=True):
                    st.code(execution, language="text", line_numbers=True)

            raw_execution = getattr(fb, "raw_execution", "")
            if raw_execution:
                with st.expander("Full Docker Log", expanded=False):
                    st.code(raw_execution, language="text", line_numbers=True)

            return_checking = getattr(fb, "return_checking", "")
            if return_checking:
                with st.expander("Return Checking", expanded=False):
                    st.code(return_checking, language="text", line_numbers=True)

            code_fb = getattr(fb, "code", "")
            if code_fb:
                st.markdown("**Code Feedback:**")
                st.markdown(code_fb)
        return

    # FTExperiment (runner result)
    if hasattr(content, "sub_workspace_list"):
        for ws in content.sub_workspace_list:
            if not hasattr(ws, "running_info") or ws.running_info is None:
                continue

            info = ws.running_info
            running_time = getattr(info, "running_time", None)
            if running_time:
                st.metric("Running Time", f"{running_time:.1f}s")

            stdout = getattr(info, "stdout", "")
            if stdout:
                with st.expander("Full Train Log", expanded=True):
                    st.code(stdout, language="text", line_numbers=True)

            result = getattr(info, "result", {})
            if result:
                render_training_result(result)
        return

    st.json(content) if content else st.info("No content")


def render_feedback(content: Any) -> None:
    col1, col2 = st.columns(2)
    with col1:
        decision = getattr(content, "decision", None)
        if decision is not None:
            st.metric("Decision", "Accept" if decision else "Reject")
    with col2:
        acceptable = getattr(content, "acceptable", None)
        if acceptable is not None:
            st.metric("Acceptable", "Yes" if acceptable else "No")

    fields = [
        ("code_change_summary", "Code Change Summary"),
        ("observations", "Observations"),
        ("hypothesis_evaluation", "Hypothesis Evaluation"),
        ("new_hypothesis", "New Hypothesis"),
        ("eda_improvement", "EDA Improvement"),
    ]

    for attr, label in fields:
        value = getattr(content, attr, None)
        if value:
            with st.expander(label, expanded=False):
                st.markdown(value)

    reason = getattr(content, "reason", None)
    if reason:
        with st.expander("Reason (Full Details)", expanded=True):
            st.code(reason, language="text", line_numbers=True)

    exception = getattr(content, "exception", None)
    if exception:
        st.error(f"Exception: {exception}")


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
        st.json(content) if content else st.info("No content")


def render_time_info(content: Any) -> None:
    if isinstance(content, dict):
        for k, v in content.items():
            st.metric(k, f"{v:.1f}s" if isinstance(v, (int, float)) else str(v))
    else:
        st.json(content) if content else st.info("No content")


def render_generic(content: Any) -> None:
    if hasattr(content, "__dict__"):
        st.json(vars(content))
    elif content:
        st.json(content)
    else:
        st.info("No content")


def render_training_result(result: dict) -> None:
    training_metrics = result.get("training_metrics", {})
    loss_history = training_metrics.get("loss_history", [])

    if loss_history:
        fig = go.Figure()
        steps = [entry.get("step", i) for i, entry in enumerate(loss_history)]
        losses = [entry.get("loss", 0) for entry in loss_history]
        fig.add_trace(go.Scatter(x=steps, y=losses, mode="lines+markers", name="Loss"))
        fig.update_layout(title="Training Loss", xaxis_title="Step", yaxis_title="Loss", height=300)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        initial_loss = training_metrics.get("initial_loss")
        final_loss = training_metrics.get("final_loss")
        if initial_loss:
            col1.metric("Initial Loss", f"{initial_loss:.4f}")
        if final_loss:
            col2.metric("Final Loss", f"{final_loss:.4f}")

    benchmark = result.get("benchmark", {})
    if benchmark:
        st.markdown("**Benchmark Results:**")
        accuracy_summary = benchmark.get("accuracy_summary", [])
        if accuracy_summary:
            st.dataframe(accuracy_summary)


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
