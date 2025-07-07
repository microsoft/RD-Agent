"""
This module provides some useful functions for working with logger folders.
"""

import pickle
from datetime import timedelta
from pathlib import Path

import pandas as pd

from rdagent.utils.workflow import LoopBase


def get_first_session_file_after_duration(log_folder: str | Path, duration: str | pd.Timedelta) -> Path:
    log_folder = Path(log_folder)
    duration_dt = pd.Timedelta(duration)
    # iterate the dump steps in increasing order
    files = sorted(
        (log_folder / "__session__").glob("*/*_*"), key=lambda f: (int(f.parent.name), int(f.name.split("_")[0]))
    )
    fp = None
    for fp in files:
        with fp.open("rb") as f:
            session_obj: LoopBase = pickle.load(f)
        timer = session_obj.timer
        all_duration = timer.all_duration
        remain_time_duration = timer.remain_time()
        if all_duration is None or remain_time_duration is None:
            msg = "Timer is not configured"
            raise ValueError(msg)
        time_spent = all_duration - remain_time_duration
        if time_spent >= duration_dt:
            break
    if fp is None:
        msg = f"No session file found after duration {duration}"
        raise ValueError(msg)
    return fp


def first_li_si_after_one_time(log_path: Path, hours: int = 12) -> tuple[int, int, str]:
    """
    Based on the hours, find the stop loop id and step id (the first step after <hours> hours).
    Args:
        log_path (Path): The path to the log folder (contains many log traces).
        hours (int): The number of hours to stat.
    Returns:
        tuple[int, int, str]: The loop id, step id and function name.
    """
    session_path = log_path / "__session__"
    max_li = max(int(p.name) for p in session_path.iterdir() if p.is_dir() and p.name.isdigit())
    max_step = max(int(p.name.split("_")[0]) for p in (session_path / str(max_li)).iterdir() if p.is_file())
    rdloop_obj_p = next((session_path / str(max_li)).glob(f"{max_step}_*"))

    rdloop_obj = DataScienceRDLoop.load(rdloop_obj_p)
    loop_trace = rdloop_obj.loop_trace
    si2fn = rdloop_obj.steps

    duration = timedelta(seconds=0)
    for li, lts in loop_trace.items():
        for lt in lts:
            si = lt.step_idx
            duration += lt.end - lt.start
            if duration > timedelta(hours=hours):
                return li, si, si2fn[si]


if __name__ == "__main__":
    from rdagent.app.data_science.loop import DataScienceRDLoop

    f = get_first_session_file_after_duration("<path to log aptos2019-blindness-detection>", pd.Timedelta("12h"))

    with f.open("rb") as f:
        session_obj: LoopBase = pickle.load(f)
    loop_trace = session_obj.loop_trace
    last_loop = loop_trace[max(loop_trace.keys())]
    last_step = last_loop[-1]
    session_obj.steps[last_step.step_idx]
