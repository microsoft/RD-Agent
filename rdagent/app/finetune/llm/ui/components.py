"""
FT UI Components - Hierarchical Event Renderers
"""

import re
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from rdagent.app.finetune.llm.ui.benchmarks import get_core_metric_score
from rdagent.app.finetune.llm.ui.config import ICONS
from rdagent.app.finetune.llm.ui.data_loader import Event, EvoLoop, Loop, Session


def convert_latex_for_streamlit(text: str) -> str:
    """Convert LaTeX syntax to Streamlit-compatible format.

    Streamlit uses $...$ and $$...$$ for LaTeX rendering.
    This converts \(...\) and \[...\] to the Streamlit format.
    """
    if not text:
        return text
    # Convert \(...\) to $...$
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    # Convert \[...\] to $$...$$
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    return text


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
    """Render a single loop with lazy loading"""
    # 1. Coding stage results
    evo_results = []
    for evo in loop.coding.values():
        if evo.success is True:
            evo_results.append("✓")
        elif evo.success is False:
            evo_results.append("✗")
    coding_str = f"💻{''.join(evo_results)}" if evo_results else ""

    # 2. Running stage results
    runner_success = None
    benchmark_score = None
    for event in loop.runner:
        # Docker (Full Train) result - check exit_code, not LLM evaluation
        if event.type == "docker_exec" and "Full Train" in event.title and event.success is not None:
            runner_success = event.success
        # Benchmark score - use core metric from processor
        if event.type == "feedback" and "Benchmark Result" in event.title:
            content = event.content
            if isinstance(content, dict):
                benchmark_name = content.get("benchmark_name", "")
                accuracy_summary = content.get("accuracy_summary", {})
                if isinstance(accuracy_summary, dict) and accuracy_summary:
                    result = get_core_metric_score(benchmark_name, accuracy_summary)
                    if result is not None:
                        _, benchmark_score, _ = result

    # 3. Get feedback decision for benchmark score coloring
    feedback_decision = None
    for event in loop.feedback:
        if event.type == "feedback" and "Feedback:" in event.title:
            feedback_decision = event.success
            break

    # 4. Build title string (only show existing stages)
    parts = []
    if coding_str:
        parts.append(coding_str)
    if runner_success is not None:
        runner_str = "🏃✓" if runner_success else "🏃✗"
        parts.append(runner_str)
    # Show benchmark score with emoji based on feedback decision
    if benchmark_score is not None:
        if feedback_decision is True:
            parts.append(f"✅📊{benchmark_score:.2f}")
        elif feedback_decision is False:
            parts.append(f"❌📊{benchmark_score:.2f}")
        else:
            parts.append(f"📊{benchmark_score:.2f}")

    result_str = " ".join(parts) if parts else ""

    loop_key = f"loop_{loop.loop_id}_loaded"
    with st.expander(f"🔄 **Loop {loop.loop_id}** {result_str}", expanded=False):
        if not st.session_state.get(loop_key, False):
            # Lazy load: show button first
            if st.button("📥 Load Content", key=f"load_{loop.loop_id}"):
                st.session_state[loop_key] = True
                st.rerun()
        else:
            # Render actual content
            _render_loop_content(loop, show_types)


def _render_loop_content(loop: Loop, show_types: list[str]) -> None:
    """Render loop content (called after lazy load)"""
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
        "evaluator": render_docker_exec,  # Reuse docker_exec renderer for evaluator feedback
        "feedback": render_feedback,
        "token": render_token,
        "time": render_time_info,
        "settings": render_settings,
        "hypothesis": render_hypothesis,
        "dataset_selection": render_dataset_selection,
    }

    renderer = renderers.get(event.type, render_generic)
    with st.expander(title, expanded=False):
        # Pass event.title to docker_exec/evaluator renderers for context-aware labels
        if event.type in ("docker_exec", "evaluator"):
            renderer(event.content, event.title)
        else:
            renderer(event.content)


def render_scenario(content: Any) -> None:
    """Render scenario details (main info shown in page header, this shows extras)."""
    import json

    # 1. User target scenario
    if hasattr(content, "user_target_scenario") and content.user_target_scenario:
        st.markdown(f"**Target Scenario:** {content.user_target_scenario}")

    # 2. Benchmark description
    if hasattr(content, "benchmark_description") and content.benchmark_description:
        st.markdown(f"**Benchmark Description:** {content.benchmark_description}")

    # 3. Full timeout
    if hasattr(content, "real_full_timeout"):
        try:
            timeout_hours = content.real_full_timeout() / 60 / 60
            st.markdown(f"**Full Train Timeout:** {timeout_hours:.2f} hours")
        except Exception:
            pass

    # 4. Device info - formatted nicely
    if hasattr(content, "device_info") and content.device_info:
        device = content.device_info
        # Parse string to dict if needed
        if isinstance(device, str):
            try:
                device = json.loads(device)
            except json.JSONDecodeError:
                st.markdown(f"**Device:** `{device}`")
                device = None
        if isinstance(device, dict):
            parts = []
            # Runtime info
            runtime = device.get("runtime", {})
            if runtime.get("python_version"):
                parts.append(f"🐍 Python `{runtime['python_version'].split()[0]}`")
            if runtime.get("os"):
                parts.append(f"💻 {runtime['os']}")
            # GPU info
            gpu_info = device.get("gpu", {})
            gpus = gpu_info.get("gpus", [])
            if gpus:
                gpu_name = gpus[0].get("name", "Unknown")
                gpu_mem_gb = gpus[0].get("memory_total_gb", 0)
                if len(gpus) > 1:
                    parts.append(f"🎮 {len(gpus)}x {gpu_name} ({gpu_mem_gb}GB)")
                else:
                    parts.append(f"🎮 {gpu_name} ({gpu_mem_gb}GB)")
            if parts:
                st.markdown(" · ".join(parts))

    # 5. Model info (detailed specs)
    if hasattr(content, "model_info") and content.model_info:
        model_info = content.model_info
        if isinstance(model_info, dict) and model_info:
            with st.expander("Model Info", expanded=False):
                # Show key specs in a readable format
                if "specs" in model_info and model_info["specs"]:
                    st.markdown("**Specs:**")
                    st.code(model_info["specs"], language="text", wrap_lines=True)
                # Show other fields
                other_info = {k: v for k, v in model_info.items() if k != "specs" and v}
                if other_info:
                    st.json(other_info)

    # 6. Memory report (estimation based on hardware and model)
    if hasattr(content, "memory_report") and content.memory_report:
        with st.expander("Memory Estimation", expanded=False):
            st.code(content.memory_report, language="text", wrap_lines=True)


def render_dataset_selection(content: Any) -> None:
    if not isinstance(content, dict):
        st.json(content) if content else st.info("No content")
        return

    selected = content.get("selected_datasets", [])
    total = content.get("total_datasets", 0)
    reasoning = content.get("reasoning", "")

    if selected:
        st.markdown(f"**Selected ({len(selected)}/{total}):** " + ", ".join(f"`{ds}`" for ds in selected))

    if reasoning:
        with st.expander("Selection Reasoning", expanded=True):
            st.markdown(reasoning)


def render_hypothesis(content: Any) -> None:
    """Render hypothesis content (Base Model shown in page header, not here)."""
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

    # Check if markdown rendering is enabled
    render_md = st.session_state.get("render_markdown_toggle", False)

    system = content.get("system", "")
    if system:
        with st.expander("System Prompt", expanded=False):
            if render_md:
                st.markdown(system)
            else:
                st.code(system, language="text", line_numbers=True, wrap_lines=True)

    user = content.get("user", "")
    if user:
        with st.expander("User Prompt", expanded=False):
            if render_md:
                st.markdown(user)
            else:
                st.code(user, language="text", line_numbers=True, wrap_lines=True)

    resp = content.get("resp", "")
    if resp:
        st.markdown("**Response:**")
        if render_md:
            st.markdown(resp)
        elif resp.strip().startswith("{") or resp.strip().startswith("["):
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

    template = content.get("template", "")
    if template:
        with st.expander("Template", expanded=False):
            st.code(template, language="text", line_numbers=True, wrap_lines=True)

    rendered = content.get("rendered", "")
    if rendered:
        with st.expander("Rendered", expanded=True):
            st.code(rendered, language="text", line_numbers=True, wrap_lines=True)


def render_experiment(content: Any) -> None:
    """Render experiment tasks (Base Model and Datasets shown in page header, not here)."""
    if isinstance(content, list):
        for i, task in enumerate(content):
            if len(content) > 1:
                st.markdown(f"**Task {i}**")

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
                st.code(code, language=lang, line_numbers=True, wrap_lines=True)


def _extract_evaluator_name(title: str) -> str:
    """Extract evaluator name from event title like 'Eval (Data Processing) ✓'."""
    match = re.search(r"\(([^)]+)\)", title)
    return match.group(1) if match else ""


def _render_single_feedback(fb: Any, evaluator_name: str = "") -> None:
    """Render a single CoSTEERSingleFeedback object.

    Structure:
    - execution: LLM-generated execution summary (what happened, success/failure reason)
    - raw_execution: Raw script stdout/stderr output
    - return_checking: LLM-generated data quality assessment
    - code: LLM-generated code improvement suggestions
    """
    decision = getattr(fb, "final_decision", None)
    if decision is True:
        st.success("Execution: PASS")
    elif decision is False:
        st.error("Execution: FAIL")

    # 1. Execution Summary (LLM-generated)
    execution = getattr(fb, "execution", "")
    if execution:
        label = f"{evaluator_name} Summary" if evaluator_name else "Execution Summary"
        with st.expander(label, expanded=True):
            st.code(execution, language="text", line_numbers=True, wrap_lines=True)

    # 2. Raw Execution Log (script stdout)
    raw_execution = getattr(fb, "raw_execution", "")
    if raw_execution:
        with st.expander("Raw Output (stdout)", expanded=False):
            st.code(raw_execution, language="text", line_numbers=True, wrap_lines=True)

    # 3. Data Quality Check (LLM-generated)
    return_checking = getattr(fb, "return_checking", "")
    if return_checking:
        with st.expander("Data Quality Check", expanded=False):
            st.code(return_checking, language="text", line_numbers=True, wrap_lines=True)

    # 4. Code Improvement Suggestions (LLM-generated, often very long)
    code_fb = getattr(fb, "code", "")
    if code_fb:
        with st.expander("Code Improvement Suggestions", expanded=False):
            # Use markdown rendering if content contains markdown formatting
            if "**" in code_fb or "```" in code_fb or "- " in code_fb:
                st.markdown(code_fb)
            else:
                st.code(code_fb, language="text", line_numbers=True, wrap_lines=True)


def render_docker_exec(content: Any, event_title: str = "") -> None:
    # Extract evaluator name from event title for context-aware labels
    evaluator_name = _extract_evaluator_name(event_title)

    # Docker run raw output (dict with exit_code/stdout)
    if isinstance(content, dict) and ("exit_code" in content or "stdout" in content or "success" in content):
        # Show workspace ID if available (only the UUID part)
        workspace_path = content.get("workspace_path")
        if workspace_path:
            workspace_id = Path(workspace_path).name
            st.caption(f"📁 `{workspace_id}`")

        exit_code = content.get("exit_code")
        success = content.get("success")
        if exit_code is not None:
            if exit_code == 0:
                st.success(f"Exit code: {exit_code}")
            else:
                st.error(f"Exit code: {exit_code}")
        elif success is not None:
            if success:
                st.success("Execution: PASS")
            else:
                st.error("Execution: FAIL")

        stdout = content.get("stdout", "")
        if stdout:
            label = f"{evaluator_name} Output" if evaluator_name else "Execution Output"
            with st.expander(label, expanded=True):
                st.code(stdout, language="text", line_numbers=True, wrap_lines=True)
        return

    # CoSTEERMultiFeedback (has feedback_list)
    if hasattr(content, "feedback_list"):
        for i, fb in enumerate(content.feedback_list):
            if len(content.feedback_list) > 1:
                st.markdown(f"**Feedback {i}**")
            _render_single_feedback(fb, evaluator_name)
        return

    # Single CoSTEERSingleFeedback (has final_decision)
    if hasattr(content, "final_decision"):
        _render_single_feedback(content, evaluator_name)
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
                    st.code(stdout, language="text", line_numbers=True, wrap_lines=True)

            result = getattr(info, "result", {})
            if result:
                render_training_result(result)
        return

    st.json(content) if content else st.info("No content")


def render_feedback(content: Any) -> None:
    # Handle benchmark result (dict with accuracy_summary)
    if isinstance(content, dict) and "accuracy_summary" in content:
        render_benchmark_result(content)
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        decision = getattr(content, "decision", None)
        if decision is not None:
            st.metric("Decision", "Accept" if decision else "Reject")
    with col2:
        acceptable = getattr(content, "acceptable", None)
        if acceptable is not None:
            st.metric("Acceptable", "Yes" if acceptable else "No")
    with col3:
        error_type = getattr(content, "observations", None)
        if error_type:
            st.metric("Error Type", error_type)

    # FT scenario only uses code_change_summary (observations, hypothesis_evaluation,
    # new_hypothesis, eda_improvement are DS scenario specific)
    fields = [
        ("code_change_summary", "Code Change Summary"),
    ]

    for attr, label in fields:
        value = getattr(content, attr, None)
        if value:
            with st.expander(label, expanded=False):
                st.markdown(value)

    reason = getattr(content, "reason", None)
    if reason:
        with st.expander("Reason (Full Details)", expanded=True):
            st.code(reason, language="text", line_numbers=True, wrap_lines=True)

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
    loss_history = training_metrics.get("loss_history", {})

    # loss_history is Dict[str, List[Dict]] with "train" and "eval" keys
    train_history = loss_history.get("train", []) if isinstance(loss_history, dict) else []
    if train_history:
        fig = go.Figure()
        steps = [entry.get("step", i) for i, entry in enumerate(train_history)]
        losses = [entry.get("loss", 0) for entry in train_history]
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

    # Validation benchmark ([:100]) - used for SOTA judgment
    benchmark = result.get("benchmark", {})
    if benchmark:
        st.markdown("**Validation Benchmark**")
        # Detect format: old format has "accuracy_summary" at top level,
        # new format has benchmark names as keys with nested accuracy_summary
        if "accuracy_summary" in benchmark:
            # Old format: {accuracy_summary: {...}, error_samples: [...]}
            accuracy_summary = benchmark.get("accuracy_summary", {})
            if accuracy_summary:
                rows = [{"dataset": ds, **metrics} for ds, metrics in accuracy_summary.items()]
                st.dataframe(rows)
        else:
            # New format: {bm_name: {accuracy_summary: {...}}, ...}
            for bm_name, bm_result in benchmark.items():
                if isinstance(bm_result, dict) and "accuracy_summary" in bm_result:
                    st.markdown(f"*{bm_name}:*")
                    accuracy_summary = bm_result.get("accuracy_summary", {})
                    if accuracy_summary:
                        rows = [{"dataset": ds, **metrics} for ds, metrics in accuracy_summary.items()]
                        st.dataframe(rows)

    # Test benchmark ([100:200]) - frontend display only, not visible to agent
    benchmark_test = result.get("benchmark_test", {})
    if benchmark_test and benchmark_test != benchmark:  # Avoid duplicate display for small datasets
        st.markdown("**Test Benchmark**")
        if "accuracy_summary" in benchmark_test:
            accuracy_summary = benchmark_test.get("accuracy_summary", {})
            if accuracy_summary:
                rows = [{"dataset": ds, **metrics} for ds, metrics in accuracy_summary.items()]
                st.dataframe(rows)
        else:
            for bm_name, bm_result in benchmark_test.items():
                if isinstance(bm_result, dict) and "accuracy_summary" in bm_result:
                    st.markdown(f"*{bm_name}:*")
                    accuracy_summary = bm_result.get("accuracy_summary", {})
                    if accuracy_summary:
                        rows = [{"dataset": ds, **metrics} for ds, metrics in accuracy_summary.items()]
                        st.dataframe(rows)


def render_benchmark_result(content: dict) -> None:
    """Render benchmark evaluation result"""
    import pandas as pd

    benchmark_name = content.get("benchmark_name", "Unknown")
    st.markdown(f"**Benchmark: {benchmark_name}**")

    # Accuracy summary table
    # accuracy_summary is a dict: {dataset_name: {metric: value, ...}, ...}
    accuracy_summary = content.get("accuracy_summary", {})
    if accuracy_summary and isinstance(accuracy_summary, dict):
        st.markdown("**Accuracy Summary:**")
        # Convert dict {dataset: {metric: value}} to list of dicts for dataframe
        rows = []
        for ds, metrics in accuracy_summary.items():
            row = {"dataset": ds, **metrics}
            rows.append(row)

        # Create DataFrame and reorder columns
        df = pd.DataFrame(rows)
        cols = ["dataset"] + [c for c in df.columns if c != "dataset"]
        df = df[cols]
        st.dataframe(df)

    # Error samples
    error_samples = content.get("error_samples", [])
    if error_samples:
        with st.expander(f"Error Samples ({len(error_samples)})", expanded=False):
            for i, sample in enumerate(error_samples):
                with st.expander(f"Sample {i+1} (Gold: {sample.get('gold', 'N/A')})", expanded=False):
                    st.markdown(
                        '<div style="font-size: 0.85em;">',
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Question:**")
                    st.markdown(convert_latex_for_streamlit(sample.get("question", "N/A")))
                    st.markdown("---")
                    st.markdown(f"**Gold:** `{sample.get('gold', 'N/A')}`")
                    st.markdown("---")
                    st.markdown("**Model Output:**")
                    st.markdown(convert_latex_for_streamlit(sample.get("model_output", "N/A")))
                    st.markdown("</div>", unsafe_allow_html=True)


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
        st.metric("Executions", f"{success}✓ / {fail}✗")
