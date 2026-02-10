"""
RL Job Summary View
Display summary table for all tasks in a job directory
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


def is_valid_task(task_path: Path) -> bool:
    """Check if directory is a valid RL task (has __session__ subdirectory)"""
    return task_path.is_dir() and (task_path / "__session__").exists()


def get_loop_dirs(task_path: Path) -> list[Path]:
    """Get sorted list of Loop directories"""
    loops = [d for d in task_path.iterdir() if d.is_dir() and d.name.startswith("Loop_")]
    return sorted(loops, key=lambda d: int(d.name.split("_")[1]))


def get_loop_status(task_path: Path, loop_id: int) -> tuple[str, bool | None]:
    """
    Get loop status and feedback decision.
    Returns: (status_str, feedback_decision)
    Status: 'C'=Coding, 'R'=Running, 'X'=Failed, 'OK'=Success
    """
    loop_path = task_path / f"Loop_{loop_id}"
    if not loop_path.exists():
        return "-", None

    # Check for feedback
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

    if feedback_decision is not None:
        return ("OK" if feedback_decision else "X"), feedback_decision

    # Check running stage
    running_files = list(loop_path.rglob("**/running/**/*.pkl"))
    if running_files:
        return "R", None

    # Check coding stage
    coding_files = list(loop_path.rglob("**/coding/**/*.pkl"))
    if coding_files:
        return "C", None

    return "?", None


def get_max_loops(job_path: Path) -> int:
    """Get maximum number of loops across all tasks"""
    max_loops = 0
    for task_dir in job_path.iterdir():
        if is_valid_task(task_dir):
            loops = get_loop_dirs(task_dir)
            max_loops = max(max_loops, len(loops))
    return max_loops


def get_job_summary_df(job_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate summary DataFrame for all tasks in job"""
    if not job_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    tasks = [d for d in sorted(job_path.iterdir(), reverse=True) if is_valid_task(d)]
    if not tasks:
        return pd.DataFrame(), pd.DataFrame()

    max_loops = get_max_loops(job_path)
    if max_loops == 0:
        max_loops = 10

    data = []
    decisions_data = []
    for task_path in tasks:
        row = {"Task": task_path.name}
        decision_row = {"Task": task_path.name}
        success_count = 0
        fail_count = 0

        for i in range(max_loops):
            status, feedback_decision = get_loop_status(task_path, i)
            row[f"L{i}"] = status
            decision_row[f"L{i}"] = feedback_decision
            if feedback_decision is True:
                success_count += 1
            elif feedback_decision is False:
                fail_count += 1

        row["Summary"] = f"{success_count}✓/{fail_count}✗"
        decision_row["Summary"] = None
        data.append(row)
        decisions_data.append(decision_row)

    df = pd.DataFrame(data)
    decisions_df = pd.DataFrame(decisions_data)
    if not df.empty:
        loop_cols = [c for c in df.columns if c.startswith("L")]
        cols = ["Task"] + sorted(loop_cols, key=lambda x: int(x[1:])) + ["Summary"]
        df = df[cols]
        decisions_df = decisions_df[cols]
    return df, decisions_df


def style_status_cell(val: str, decision: bool | None = None) -> str:
    """Style cell based on status value"""
    if val == "-":
        return "color: #888"
    if val == "C":
        return "color: #f0ad4e; font-weight: bold"
    if val == "R":
        return "color: #5bc0de; font-weight: bold"
    if val == "X":
        return "color: #d9534f; font-weight: bold"
    if val == "OK":
        return "color: #5cb85c; font-weight: bold"
    if val == "?":
        return "color: #888"
    return ""


def style_df_with_decisions(df: pd.DataFrame, decisions_df: pd.DataFrame):
    """Apply styling to dataframe"""
    def apply_styles(row_idx: int, col: str) -> str:
        val = df.iloc[row_idx][col]
        decision = decisions_df.iloc[row_idx][col] if col in decisions_df.columns else None
        return style_status_cell(str(val), decision)

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

    st.markdown(
        "**Legend:** "
        "<span style='color:#f0ad4e'>C</span>=Coding, "
        "<span style='color:#5bc0de'>R</span>=Running, "
        "<span style='color:#5cb85c'>OK</span>=Success, "
        "<span style='color:#d9534f'>X</span>=Failed",
        unsafe_allow_html=True,
    )

    styled_df = style_df_with_decisions(df, decisions_df)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tasks", len(df))
    with col2:
        loop_cols = [c for c in decisions_df.columns if c.startswith("L")]
        tasks_success = decisions_df[loop_cols].apply(lambda row: any(v is True for v in row), axis=1).sum()
        st.metric("With Success", tasks_success)
    with col3:
        total_loops = sum(1 for _, row in decisions_df.iterrows() for c in loop_cols if row[c] is not None)
        st.metric("Total Loops", total_loops)

