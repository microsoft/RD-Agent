"""
FT 场景 UI 组件
"""

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def show_scenario_info(scenario: Any) -> None:
    """显示场景信息"""
    if scenario is None:
        st.info("No scenario info available")
        return

    st.subheader("Scenario Info")

    col1, col2 = st.columns(2)
    with col1:
        if hasattr(scenario, "base_model"):
            st.metric("Base Model", scenario.base_model)
        if hasattr(scenario, "target_benchmark"):
            st.metric("Target Benchmark", scenario.target_benchmark)
    with col2:
        if hasattr(scenario, "dataset"):
            st.metric("Dataset", scenario.dataset)
        if hasattr(scenario, "device_info"):
            st.text(f"Device: {scenario.device_info}")


def show_hypothesis(experiment: Any) -> None:
    """显示假设信息（不显示 benchmark，显示 involving_datasets）"""
    if experiment is None:
        st.info("No experiment info available")
        return

    st.subheader("Hypothesis")

    # experiment 可能是 list[Task] 或 FTExperiment
    hypo = None
    if hasattr(experiment, "hypothesis"):
        hypo = experiment.hypothesis
    elif isinstance(experiment, list) and len(experiment) > 0:
        task = experiment[0]
        if hasattr(task, "task_type"):
            # FTTask - 显示 task_type, base_model, involving_datasets
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Task Type", getattr(task, "task_type", "N/A"))
                st.metric("Base Model", getattr(task, "base_model", "N/A"))
            with col2:
                # 显示 involving_datasets
                datasets = getattr(task, "involving_datasets", None)
                if datasets:
                    st.markdown("**Involving Datasets:**")
                    for ds in datasets:
                        st.markdown(f"- `{ds}`")
                else:
                    st.markdown("**Involving Datasets:** None")

            if hasattr(task, "description"):
                with st.expander("Task Description", expanded=True):
                    st.markdown(task.description)
            return

    if hypo is None:
        st.info("No hypothesis available")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Task Type", getattr(hypo, "task_type", "N/A"))
        st.metric("Base Model", getattr(hypo, "base_model", "N/A"))
    with col2:
        if hasattr(hypo, "hypothesis") and hypo.hypothesis:
            st.markdown(f"**Hypothesis:** {hypo.hypothesis}")
        if hasattr(hypo, "reason") and hypo.reason:
            st.markdown(f"**Reason:** {hypo.reason}")


def show_files(workspace_list: list[Any] | None) -> None:
    """显示生成的代码文件"""
    st.subheader("Generated Files")

    if not workspace_list:
        st.info("No workspace available")
        return

    for i, ws in enumerate(workspace_list):
        if not hasattr(ws, "file_dict") or not ws.file_dict:
            continue

        if len(workspace_list) > 1:
            st.markdown(f"**Workspace {i}**")

        for filename, content in ws.file_dict.items():
            lang = "yaml" if filename.endswith((".yaml", ".yml")) else "python"
            with st.expander(f"{filename}"):
                st.code(content, language=lang, line_numbers=True)


def show_evo_loops(evo_loops: dict[int, dict]) -> None:
    """显示演化循环信息（包含完整的 feedback）"""
    if not evo_loops:
        return

    st.subheader("Evolution Loops")

    for evo_id in sorted(evo_loops.keys()):
        evo_data = evo_loops[evo_id]
        with st.expander(f"Evo Loop {evo_id}"):
            # 代码（可折叠）
            code_list = evo_data.get("code")
            if code_list and isinstance(code_list, list):
                st.markdown("**Generated Code:**")
                for ws in code_list:
                    if hasattr(ws, "file_dict") and ws.file_dict:
                        for filename, content in ws.file_dict.items():
                            lang = "yaml" if filename.endswith((".yaml", ".yml")) else "python"
                            with st.expander(filename):
                                st.code(content, language=lang, line_numbers=True)

            # 反馈 - 显示合并后的反馈（包含 Data Processing + Debug Train）
            feedback = evo_data.get("feedback")
            if feedback is not None and hasattr(feedback, "feedback_list"):
                st.markdown("**Feedback:**")
                for i, fb in enumerate(feedback.feedback_list):
                    decision = getattr(fb, "final_decision", None)
                    status = "Pass" if decision else "Fail" if decision is not None else "Unknown"

                    # feedback_list 通常只有 1 个元素（合并了 Data Processing + Debug Train）
                    label = "Coder Feedback" if len(feedback.feedback_list) == 1 else f"Feedback {i}"

                    with st.expander(f"{label} [{status}]"):
                        # execution（包含 Data Processing + Debug Train 的合并日志）
                        if hasattr(fb, "execution") and fb.execution:
                            st.markdown("**Execution Log (Data Processing + Debug Train):**")
                            st.code(fb.execution, language="text")

                        # return_checking
                        if hasattr(fb, "return_checking") and fb.return_checking:
                            st.markdown("**Return Checking:**")
                            st.code(fb.return_checking, language="text")

                        # code feedback
                        if hasattr(fb, "code") and fb.code:
                            st.markdown("**Code Feedback:**")
                            st.markdown(fb.code)


def show_results(workspace_list: list[Any] | None) -> None:
    """显示训练结果（Benchmark 结果）"""
    st.subheader("Results")

    if not workspace_list:
        st.info("No results available")
        return

    has_result = False
    for ws in workspace_list:
        if not hasattr(ws, "running_info") or not ws.running_info:
            continue

        result = getattr(ws.running_info, "result", None)
        if not result:
            continue

        has_result = True

        # 训练指标
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

        # Benchmark 结果
        benchmark = result.get("benchmark", {})
        if benchmark:
            st.markdown("**Benchmark Results:**")
            accuracy_summary = benchmark.get("accuracy_summary", [])
            if accuracy_summary:
                st.dataframe(accuracy_summary)

            # 错误样本
            error_samples = benchmark.get("error_samples", [])
            if error_samples:
                with st.expander(f"Error Samples ({len(error_samples)})"):
                    for sample in error_samples[:5]:
                        st.json(sample)

    if not has_result:
        st.info("No results available")


def show_feedback(feedback: Any) -> None:
    """显示最终反馈（显示所有可用字段）"""
    st.subheader("Final Feedback")

    if feedback is None:
        st.info("No feedback available")
        return

    # 基本决策信息
    col1, col2 = st.columns(2)
    with col1:
        decision = getattr(feedback, "decision", None)
        if decision is not None:
            st.metric("Decision", "Accept" if decision else "Reject")
    with col2:
        acceptable = getattr(feedback, "acceptable", None)
        if acceptable is not None:
            st.metric("Acceptable", "Yes" if acceptable else "No")

    # code_change_summary
    code_change_summary = getattr(feedback, "code_change_summary", None)
    if code_change_summary:
        st.markdown(f"**Code Change Summary:** {code_change_summary}")

    # observations
    observations = getattr(feedback, "observations", None)
    if observations:
        with st.expander("Observations"):
            st.markdown(observations)

    # hypothesis_evaluation
    hypothesis_evaluation = getattr(feedback, "hypothesis_evaluation", None)
    if hypothesis_evaluation:
        with st.expander("Hypothesis Evaluation"):
            st.markdown(hypothesis_evaluation)

    # new_hypothesis
    new_hypothesis = getattr(feedback, "new_hypothesis", None)
    if new_hypothesis:
        with st.expander("New Hypothesis"):
            st.markdown(new_hypothesis)

    # eda_improvement
    eda_improvement = getattr(feedback, "eda_improvement", None)
    if eda_improvement:
        with st.expander("EDA Improvement"):
            st.markdown(eda_improvement)

    # reason - 通常包含完整的执行日志
    reason = getattr(feedback, "reason", None)
    if reason:
        with st.expander("Reason (Full Details)", expanded=True):
            st.code(reason, language="text")

    # exception
    exception = getattr(feedback, "exception", None)
    if exception:
        st.error(f"Exception: {exception}")


def show_runner_result(runner_result: Any) -> None:
    """显示 Full Train 执行结果"""
    st.subheader("Full Train Result")

    if runner_result is None:
        st.info("No runner result available (training may have been skipped)")
        return

    # runner_result 是 FTExperiment 对象
    if not hasattr(runner_result, "sub_workspace_list"):
        st.warning("Invalid runner result format")
        return

    for i, ws in enumerate(runner_result.sub_workspace_list):
        if not hasattr(ws, "running_info") or ws.running_info is None:
            continue

        running_info = ws.running_info

        # 显示执行时间
        running_time = getattr(running_info, "running_time", None)
        if running_time:
            st.metric("Running Time", f"{running_time:.1f}s")

        # 显示执行日志
        stdout = getattr(running_info, "stdout", None)
        if stdout:
            with st.expander("Full Train Execution Log", expanded=True):
                st.code(stdout, language="text")

        # 显示结果摘要
        result = getattr(running_info, "result", None)
        if result:
            # 训练是否成功
            training_metrics = result.get("training_metrics", {})
            if training_metrics:
                final_loss = training_metrics.get("final_loss")
                if final_loss:
                    st.metric("Final Loss", f"{final_loss:.4f}")
