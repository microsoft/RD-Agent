"""
FT Job Summary View
Display summary table for all tasks in a job directory
"""

import pickle
from pathlib import Path

import pandas as pd
from pandas.io.formats.style import Styler
import streamlit as st

from rdagent.app.finetune.llm.ui.benchmarks import get_core_metric_score


def is_valid_task(task_path: Path) -> bool:
    """Check if directory is a valid FT task (has __session__ subdirectory)"""
    return task_path.is_dir() and (task_path / "__session__").exists()


def get_loop_dirs(task_path: Path) -> list[Path]:
    """Get sorted list of Loop directories"""
    loops = [d for d in task_path.iterdir() if d.is_dir() and d.name.startswith("Loop_")]
    return sorted(loops, key=lambda d: int(d.name.split("_")[1]))


def extract_benchmark_score(loop_path: Path, split: str = "") -> tuple[str, float, bool] | None:
    """Extract benchmark score, metric name, and direction from loop directory.

    Args:
        loop_path: Path to loop directory
        split: Filter by split type ("validation", "test", or "" for any)

    Returns:
        (metric_name, score, higher_is_better) or None
        - metric_name includes "(average)" suffix if multiple datasets are averaged
        - higher_is_better: True if higher values are better
    """
    for pkl_file in loop_path.rglob("**/benchmark_result*/**/*.pkl"):
        try:
            with open(pkl_file, "rb") as f:
                content = pickle.load(f)
            if isinstance(content, dict):
                # Check split filter
                content_split = content.get("split", "")
                if split and content_split != split:
                    continue

                benchmark_name = content.get("benchmark_name", "")
                accuracy_summary = content.get("accuracy_summary", {})
                if isinstance(accuracy_summary, dict) and accuracy_summary:
                    result = get_core_metric_score(benchmark_name, accuracy_summary)
                    if result is not None:
                        return result
        except Exception:
            pass
    return None


def extract_benchmark_scores(loop_path: Path) -> dict[str, tuple[str, float, bool] | None]:
    """Extract both validation and test benchmark scores from loop directory.

    Returns:
        Dict with keys "validation" and "test", each containing
        (metric_name, score, higher_is_better) or None
    """
    return {
        "validation": extract_benchmark_score(loop_path, split="validation"),
        "test": extract_benchmark_score(loop_path, split="test"),
    }


def extract_baseline_score(task_path: Path) -> tuple[str, float] | None:
    """Extract baseline benchmark score from scenario object (legacy, validation only).

    Returns:
        (metric_name, score) or None
    """
    scenario_dir = task_path / "scenario"
    if not scenario_dir.exists():
        return None

    for pkl_file in scenario_dir.rglob("*.pkl"):
        try:
            with open(pkl_file, "rb") as f:
                scenario = pickle.load(f)
            baseline_score = getattr(scenario, "baseline_benchmark_score", None)
            if baseline_score and isinstance(baseline_score, dict):
                benchmark_name = getattr(scenario, "target_benchmark", "")
                accuracy_summary = baseline_score.get("accuracy_summary", {})
                if isinstance(accuracy_summary, dict) and accuracy_summary:
                    result = get_core_metric_score(benchmark_name, accuracy_summary)
                    if result is not None:
                        metric_name, score, _ = result
                        return metric_name, score
        except Exception:
            pass
    return None


def extract_baseline_scores(task_path: Path) -> dict[str, tuple[str, float, bool] | None]:
    """Extract both validation and test baseline benchmark scores from scenario.

    Returns:
        {"validation": (metric_name, score, higher_is_better) or None,
         "test": (metric_name, score, higher_is_better) or None}
    """
    scenario_dir = task_path / "scenario"
    if not scenario_dir.exists():
        return {"validation": None, "test": None}

    for pkl_file in scenario_dir.rglob("*.pkl"):
        try:
            with open(pkl_file, "rb") as f:
                scenario = pickle.load(f)

            benchmark_name = getattr(scenario, "target_benchmark", "")
            result = {"validation": None, "test": None}

            # Validation score
            baseline_val = getattr(scenario, "baseline_benchmark_score", None)
            if baseline_val and isinstance(baseline_val, dict):
                accuracy_summary = baseline_val.get("accuracy_summary", {})
                if isinstance(accuracy_summary, dict) and accuracy_summary:
                    core = get_core_metric_score(benchmark_name, accuracy_summary)
                    if core:
                        result["validation"] = core

            # Test score (new format only)
            baseline_test = getattr(scenario, "baseline_benchmark_score_test", None)
            if baseline_test and isinstance(baseline_test, dict):
                accuracy_summary = baseline_test.get("accuracy_summary", {})
                if isinstance(accuracy_summary, dict) and accuracy_summary:
                    core = get_core_metric_score(benchmark_name, accuracy_summary)
                    if core:
                        result["test"] = core

            return result
        except Exception:
            pass
    return {"validation": None, "test": None}


def get_loop_status(
    task_path: Path, loop_id: int
) -> tuple[str, float | None, float | None, str | None, bool | None, bool]:
    """
    Get loop status, validation score, test score, metric name with direction arrow, feedback decision, and direction.
    Returns: (status_str, val_score_or_none, test_score_or_none, metric_display_or_none, feedback_decision, higher_is_better)
    Status: 'C'=Coding, 'R'=Running, 'X'=Failed, score_str=Success
    metric_display: metric name with direction arrow (e.g., "accuracy ↑")
    feedback_decision: True=accepted, False=rejected, None=no feedback
    higher_is_better: True if higher values are better for this metric
    """
    loop_path = task_path / f"Loop_{loop_id}"
    if not loop_path.exists():
        return "-", None, None, None, None, True

    # Check for benchmark results first (highest priority - means completed)
    scores = extract_benchmark_scores(loop_path)
    val_result = scores.get("validation")
    test_result = scores.get("test")

    # Fallback to old format (no split) if no validation/test found
    if val_result is None and test_result is None:
        legacy_result = extract_benchmark_score(loop_path, split="")
        if legacy_result is not None:
            val_result = legacy_result  # Treat legacy as validation

    # Get feedback decision (used for both score coloring and fallback status)
    feedback_decision = None
    feedback_files = list(loop_path.rglob("**/feedback/**/*.pkl"))
    for f in feedback_files:
        try:
            with open(f, "rb") as fp:
                content = pickle.load(fp)
            decision = getattr(content, "decision", None)
            if decision is not None:
                feedback_decision = decision
                break
        except Exception:
            pass

    if val_result is not None:
        metric_name, val_score, higher_is_better = val_result
        test_score = test_result[1] if test_result else None
        arrow = "↑" if higher_is_better else "↓"
        metric_display = f"{metric_name} {arrow}"
        # Format: "val/test" or just "val" if no test
        if test_score is not None:
            status_str = f"{val_score:.2f}/{test_score:.2f}"
        else:
            status_str = f"{val_score:.2f}"
        return status_str, val_score, test_score, metric_display, feedback_decision, higher_is_better

    # Check feedback stage (no benchmark result, use feedback decision directly)
    if feedback_decision is not None:
        return ("OK" if feedback_decision else "X"), None, None, None, feedback_decision, True

    # Check running stage
    running_files = list(loop_path.rglob("**/running/**/*.pkl"))
    if running_files:
        return "R", None, None, None, None, True

    # Check coding stage
    coding_files = list(loop_path.rglob("**/coding/**/*.pkl"))
    if coding_files:
        return "C", None, None, None, None, True

    # Has directory but no recognized files
    return "?", None, None, None, None, True


def get_max_loops(job_path: Path) -> int:
    """Get maximum number of loops across all tasks"""
    max_loops = 0
    for task_dir in job_path.iterdir():
        if is_valid_task(task_dir):
            loops = get_loop_dirs(task_dir)
            max_loops = max(max_loops, len(loops))
    return max_loops


def get_job_summary_df(job_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate summary DataFrame and decision DataFrame for all tasks in job

    Each loop column shows "val/test" format when both scores are available.
    Best columns show the best validation and test scores separately.

    Returns:
        (df, decisions_df): df is display data, decisions_df has same structure
        but values are True/False/None for feedback decision
    """
    if not job_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    tasks = [d for d in sorted(job_path.iterdir(), reverse=True) if is_valid_task(d)]
    if not tasks:
        return pd.DataFrame(), pd.DataFrame()

    max_loops = get_max_loops(job_path)
    if max_loops == 0:
        max_loops = 10  # Default display columns

    data = []
    decisions_data = []
    for task_path in tasks:
        row = {"Task": task_path.name}
        decision_row = {"Task": task_path.name}
        best_val_score = None
        best_test_score = None
        best_metric = None
        best_higher_is_better = True  # Default to higher is better

        # Extract baseline scores (validation and test) from scenario
        baseline_scores = extract_baseline_scores(task_path)
        val_baseline = baseline_scores.get("validation")
        test_baseline = baseline_scores.get("test")
        if val_baseline and test_baseline:
            row["Baseline"] = f"{val_baseline[1]:.2f}/{test_baseline[1]:.2f}"
        elif val_baseline:
            row["Baseline"] = f"{val_baseline[1]:.2f}"
        else:
            row["Baseline"] = "-"
        decision_row["Baseline"] = None

        for i in range(max_loops):
            status, val_score, test_score, metric_name, feedback_decision, higher_is_better = get_loop_status(
                task_path, i
            )
            row[f"L{i}"] = status
            decision_row[f"L{i}"] = feedback_decision
            if val_score is not None:
                # Use higher_is_better to determine if this score is better
                if best_val_score is None:
                    best_val_score = val_score
                    best_higher_is_better = higher_is_better
                    best_metric = metric_name
                elif (higher_is_better and val_score > best_val_score) or (
                    not higher_is_better and val_score < best_val_score
                ):
                    best_val_score = val_score
                    best_higher_is_better = higher_is_better
                    best_metric = metric_name
            if test_score is not None:
                # Use same direction as validation score for consistency
                if best_test_score is None:
                    best_test_score = test_score
                elif (best_higher_is_better and test_score > best_test_score) or (
                    not best_higher_is_better and test_score < best_test_score
                ):
                    best_test_score = test_score

        # Show best validation and test scores
        if best_val_score is not None and best_test_score is not None:
            row["Best"] = f"{best_val_score:.2f}/{best_test_score:.2f}"
        elif best_val_score is not None:
            row["Best"] = f"{best_val_score:.2f}"
        else:
            row["Best"] = "-"
        row["Metric"] = best_metric if best_metric else "-"
        decision_row["Metric"] = None
        decision_row["Best"] = None
        data.append(row)
        decisions_data.append(decision_row)

    # Ensure column order: Task, Metric, Baseline, L0, L1, ..., Best
    df = pd.DataFrame(data)
    decisions_df = pd.DataFrame(decisions_data)
    if not df.empty:
        loop_cols = [c for c in df.columns if c.startswith("L")]
        cols = ["Task", "Metric", "Baseline"] + sorted(loop_cols, key=lambda x: int(x[1:])) + ["Best"]
        df = df[cols]
        decisions_df = decisions_df[cols]
    return df, decisions_df


def style_status_cell(val: str, decision: bool | None = None) -> str:
    """Style cell based on status value and feedback decision

    Args:
        val: The cell value
        decision: True=accepted (green), False=rejected (red), None=no feedback (gray)
    """
    if val == "-":
        return "color: #888"
    if val == "C":
        return "color: #f0ad4e; font-weight: bold"  # Orange for coding
    if val == "R":
        return "color: #5bc0de; font-weight: bold"  # Blue for running
    if val == "X":
        return "color: #d9534f; font-weight: bold"  # Red for failed
    if val == "OK":
        return "color: #5cb85c; font-weight: bold"  # Green for success
    if val == "?":
        return "color: #888"

    # Check if it's a numeric score (with optional "/" separator)
    is_numeric = False
    try:
        float(val)
        is_numeric = True
    except ValueError:
        if "/" in val:
            parts = val.split("/")
            try:
                float(parts[0])
                is_numeric = True
            except ValueError:
                pass

    if is_numeric:
        # Use decision for coloring (use == instead of is for numpy.bool_ compatibility)
        if decision == True:
            return "color: #5cb85c; font-weight: bold"  # Green for accepted
        elif decision == False:
            return "color: #d9534f; font-weight: bold"  # Red for rejected
        else:
            return "color: #888"  # Gray for no feedback

    return ""


def style_df_with_decisions(df: pd.DataFrame, decisions_df: pd.DataFrame) -> Styler:
    """Apply styling to dataframe based on decision data

    Args:
        df: Display dataframe
        decisions_df: DataFrame with same shape, containing True/False/None values
    """

    def apply_styles(row_idx: int, col: str) -> str:
        val = df.iloc[row_idx][col]
        decision = decisions_df.iloc[row_idx][col] if col in decisions_df.columns else None
        return style_status_cell(str(val), decision)

    # Build style matrix
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for row_idx in range(len(df)):
        for col in df.columns:
            styles.iloc[row_idx][col] = apply_styles(row_idx, col)

    return df.style.apply(lambda _: styles, axis=None)


def render_job_summary(job_path: Path, is_root: bool = False) -> None:
    """Render job summary UI"""
    title = "Standalone Tasks" if is_root else f"Job: {job_path.name}"
    st.subheader(title)

    df, decisions_df = get_job_summary_df(job_path)
    if df.empty:
        st.warning("No valid tasks found in this job directory")
        return

    # Display legend
    st.markdown(
        "**Legend:** "
        "<span style='color:#f0ad4e'>C</span>=Coding, "
        "<span style='color:#5bc0de'>R</span>=Running, "
        "<span style='color:#5cb85c'>Score</span>=Accepted, "
        "<span style='color:#d9534f'>Score/X</span>=Rejected/Failed, "
        "<span style='color:#888'>Score</span>=No feedback",
        unsafe_allow_html=True,
    )

    # Style and display dataframe
    styled_df = style_df_with_decisions(df, decisions_df)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tasks", len(df))
    with col2:
        # Count tasks with any score
        tasks_with_score = df["Best"].apply(lambda x: x != "-").sum()
        st.metric("With Score", tasks_with_score)
    with col3:
        # Count tasks with at least one improved loop (decision=True)
        loop_cols = [c for c in decisions_df.columns if c.startswith("L")]
        tasks_improved = decisions_df[loop_cols].apply(lambda row: any(v is True for v in row), axis=1).sum()
        st.metric("Improved", tasks_improved)

    # Detailed scores table
    render_task_detail_selector(job_path)


def extract_full_benchmark(loop_path: Path, split: str = "") -> dict | None:
    """Extract full accuracy_summary from loop directory.

    Args:
        loop_path: Path to loop directory
        split: Filter by split type ("validation", "test", or "" for any)

    Returns:
        accuracy_summary dict {dataset: {metric: value, ...}, ...} or None
    """
    for pkl_file in loop_path.rglob("**/benchmark_result*/**/*.pkl"):
        try:
            with open(pkl_file, "rb") as f:
                content = pickle.load(f)
            if isinstance(content, dict):
                # Check split filter
                content_split = content.get("split", "")
                if split and content_split != split:
                    continue

                accuracy_summary = content.get("accuracy_summary", {})
                if isinstance(accuracy_summary, dict) and accuracy_summary:
                    return accuracy_summary
        except Exception:
            pass
    return None


def extract_baseline_full_benchmark(task_path: Path, split: str = "validation") -> dict | None:
    """Extract full accuracy_summary from baseline scenario.

    Args:
        task_path: Path to task directory
        split: "validation" or "test"

    Returns:
        accuracy_summary dict or None
    """
    scenario_dir = task_path / "scenario"
    if not scenario_dir.exists():
        return None

    for pkl_file in scenario_dir.rglob("*.pkl"):
        try:
            with open(pkl_file, "rb") as f:
                scenario = pickle.load(f)

            if split == "validation":
                baseline = getattr(scenario, "baseline_benchmark_score", None)
            else:
                baseline = getattr(scenario, "baseline_benchmark_score_test", None)

            if baseline and isinstance(baseline, dict):
                accuracy_summary = baseline.get("accuracy_summary", {})
                if isinstance(accuracy_summary, dict) and accuracy_summary:
                    return accuracy_summary
        except Exception:
            pass
    return None


def get_task_full_benchmark_df(task_path: Path, split: str) -> pd.DataFrame:
    """Generate full benchmark table for a single task and split.

    Returns DataFrame with columns: Dataset, Metric, Baseline, Loop_0, Loop_1, ...
    Each row is a dataset-metric combination.
    """
    # Collect all sources (Baseline + Loops)
    sources = ["Baseline"]
    loop_dirs = sorted(
        [d for d in task_path.iterdir() if d.is_dir() and d.name.startswith("Loop_")],
        key=lambda x: int(x.name.split("_")[1]),
    )
    sources.extend([d.name for d in loop_dirs])

    # Collect all accuracy_summaries
    all_summaries = {}

    # Baseline
    baseline_summary = extract_baseline_full_benchmark(task_path, split)
    if baseline_summary:
        all_summaries["Baseline"] = baseline_summary

    # Loops
    for loop_dir in loop_dirs:
        loop_summary = extract_full_benchmark(loop_dir, split)
        if loop_summary:
            all_summaries[loop_dir.name] = loop_summary

    if not all_summaries:
        return pd.DataFrame()

    # Collect all dataset-metric combinations
    all_keys = set()
    for summary in all_summaries.values():
        for dataset, metrics in summary.items():
            if isinstance(metrics, dict):
                for metric in metrics.keys():
                    all_keys.add((dataset, metric))

    # Sort keys for consistent display
    all_keys = sorted(all_keys)

    # Build table data
    data = []
    for dataset, metric in all_keys:
        row = {"Dataset": dataset, "Metric": metric}
        for source in sources:
            summary = all_summaries.get(source, {})
            metrics_dict = summary.get(dataset, {})
            value = metrics_dict.get(metric) if isinstance(metrics_dict, dict) else None
            if value is not None:
                row[source] = f"{value:.2f}" if isinstance(value, float) else str(value)
            else:
                row[source] = "-"
        data.append(row)

    df = pd.DataFrame(data)
    # Ensure column order
    if not df.empty:
        cols = ["Dataset", "Metric"] + [s for s in sources if s in df.columns]
        df = df[cols]
    return df


def render_task_detail_selector(job_path: Path) -> None:
    """Render task selector dropdown and full benchmark tables."""
    tasks = [d for d in sorted(job_path.iterdir(), reverse=True) if is_valid_task(d)]
    if not tasks:
        return

    st.markdown("---")
    st.subheader("Detailed Benchmark Scores")

    # Task selector dropdown
    task_names = [t.name for t in tasks]
    selected_task = st.selectbox("Select Task", options=task_names, index=0, key="task_detail_selector")

    if selected_task:
        task_path = job_path / selected_task

        # Display Validation and Test tables side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Validation**")
            df_val = get_task_full_benchmark_df(task_path, "validation")
            if not df_val.empty:
                st.dataframe(df_val, use_container_width=True, hide_index=True)
            else:
                st.info("No validation scores")

        with col2:
            st.markdown("**Test**")
            df_test = get_task_full_benchmark_df(task_path, "test")
            if not df_test.empty:
                st.dataframe(df_test, use_container_width=True, hide_index=True)
            else:
                st.info("No test scores")
