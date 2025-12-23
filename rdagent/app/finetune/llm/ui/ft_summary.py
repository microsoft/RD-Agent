"""
FT Job Summary View
Display summary table for all tasks in a job directory
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


def is_valid_task(task_path: Path) -> bool:
    """Check if directory is a valid FT task (has __session__ subdirectory)"""
    return task_path.is_dir() and (task_path / "__session__").exists()


def get_loop_dirs(task_path: Path) -> list[Path]:
    """Get sorted list of Loop directories"""
    loops = [d for d in task_path.iterdir() if d.is_dir() and d.name.startswith("Loop_")]
    return sorted(loops, key=lambda d: int(d.name.split("_")[1]))


def extract_benchmark_score(loop_path: Path) -> float | None:
    """Extract benchmark score from loop directory"""
    # Look for benchmark_result pkl files
    for pkl_file in loop_path.rglob("**/benchmark_result/**/*.pkl"):
        try:
            with open(pkl_file, "rb") as f:
                content = pickle.load(f)
            if isinstance(content, dict):
                # Try to extract accuracy from accuracy_summary
                accuracy_summary = content.get("accuracy_summary", [])
                if accuracy_summary:
                    # Find first numeric value in accuracy_summary
                    if isinstance(accuracy_summary, list) and len(accuracy_summary) > 0:
                        first_row = accuracy_summary[0]
                        if isinstance(first_row, dict):
                            for k, v in first_row.items():
                                if k != "dataset" and isinstance(v, (int, float)):
                                    return float(v)
        except Exception:
            pass
    return None


def get_loop_status(task_path: Path, loop_id: int) -> tuple[str, float | None]:
    """
    Get loop status and score
    Returns: (status_str, score_or_none)
    Status: 'C'=Coding, 'R'=Running, 'X'=Failed, score_str=Success
    """
    loop_path = task_path / f"Loop_{loop_id}"
    if not loop_path.exists():
        return "-", None

    # Check for benchmark result first (highest priority - means completed)
    score = extract_benchmark_score(loop_path)
    if score is not None:
        return f"{score:.1f}", score

    # Check feedback stage
    feedback_files = list(loop_path.rglob("**/feedback/**/*.pkl"))
    if feedback_files:
        # Has feedback, check if accepted/rejected
        for f in feedback_files:
            try:
                with open(f, "rb") as fp:
                    content = pickle.load(fp)
                decision = getattr(content, "decision", None)
                if decision is not None:
                    return ("OK" if decision else "X"), None
            except Exception:
                pass

    # Check running stage
    running_files = list(loop_path.rglob("**/running/**/*.pkl"))
    if running_files:
        return "R", None

    # Check coding stage
    coding_files = list(loop_path.rglob("**/coding/**/*.pkl"))
    if coding_files:
        return "C", None

    # Has directory but no recognized files
    return "?", None


def get_max_loops(job_path: Path) -> int:
    """Get maximum number of loops across all tasks"""
    max_loops = 0
    for task_dir in job_path.iterdir():
        if is_valid_task(task_dir):
            loops = get_loop_dirs(task_dir)
            max_loops = max(max_loops, len(loops))
    return max_loops


def get_job_summary_df(job_path: Path) -> pd.DataFrame:
    """Generate summary DataFrame for all tasks in job"""
    if not job_path.exists():
        return pd.DataFrame()

    tasks = [d for d in sorted(job_path.iterdir()) if is_valid_task(d)]
    if not tasks:
        return pd.DataFrame()

    max_loops = get_max_loops(job_path)
    if max_loops == 0:
        max_loops = 10  # Default display columns

    data = []
    for task_path in tasks:
        row = {"Task": task_path.name}
        best_score = None

        for i in range(max_loops):
            status, score = get_loop_status(task_path, i)
            row[f"L{i}"] = status
            if score is not None:
                if best_score is None or score > best_score:
                    best_score = score

        row["Best"] = f"{best_score:.1f}" if best_score is not None else "-"
        data.append(row)

    return pd.DataFrame(data)


def style_status_cell(val: str) -> str:
    """Style cell based on status value"""
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
    # Numeric score
    try:
        float(val)
        return "color: #5cb85c; font-weight: bold"  # Green for scores
    except ValueError:
        return ""


def render_job_summary(job_path: Path, is_root: bool = False) -> None:
    """Render job summary UI"""
    title = "Standalone Tasks" if is_root else f"Job: {job_path.name}"
    st.subheader(title)

    df = get_job_summary_df(job_path)
    if df.empty:
        st.warning("No valid tasks found in this job directory")
        return

    # Display legend
    st.markdown(
        "**Legend:** "
        "<span style='color:#f0ad4e'>C</span>=Coding, "
        "<span style='color:#5bc0de'>R</span>=Running, "
        "<span style='color:#5cb85c'>Score/OK</span>=Success, "
        "<span style='color:#d9534f'>X</span>=Failed",
        unsafe_allow_html=True,
    )

    # Style and display dataframe
    styled_df = df.style.map(style_status_cell)
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
        # Average best score
        scores = []
        for val in df["Best"]:
            try:
                scores.append(float(val))
            except ValueError:
                pass
        avg_score = sum(scores) / len(scores) if scores else 0
        st.metric("Avg Best", f"{avg_score:.1f}" if scores else "-")
