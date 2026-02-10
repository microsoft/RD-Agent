"""
RL UI Data Loader
Load pkl logs and convert to hierarchical timeline structure
Simplified version: no EvoLoop (RL doesn't have evolution loops)
"""

import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from rdagent.app.rl.ui.config import EventType
from rdagent.log.storage import FileStorage


@dataclass
class Event:
    """Timeline event"""

    type: EventType
    timestamp: datetime
    tag: str
    title: str
    content: Any
    loop_id: int | None = None
    stage: str = ""
    duration: float | None = None
    success: bool | None = None

    @property
    def time_str(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")


@dataclass
class Loop:
    """Main loop containing stages (no EvoLoop for RL)"""

    loop_id: int
    proposal: list[Event] = field(default_factory=list)  # hypothesis generation
    coding: list[Event] = field(default_factory=list)    # code generation
    running: list[Event] = field(default_factory=list)   # docker training + benchmark
    feedback: list[Event] = field(default_factory=list)  # feedback


@dataclass
class Session:
    """Session containing init events and loops"""

    init_events: list[Event] = field(default_factory=list)
    loops: dict[int, Loop] = field(default_factory=dict)


def extract_loop_id(tag: str) -> int | None:
    match = re.search(r"Loop_(\d+)", tag)
    return int(match.group(1)) if match else None


def extract_stage(tag: str) -> str:
    if "proposal" in tag or "direct_exp_gen" in tag:
        return "proposal"
    if "coding" in tag:
        return "coding"
    if "running" in tag:
        return "running"
    if "feedback" in tag:
        return "feedback"
    return ""


def get_valid_sessions(log_folder: Path) -> list[str]:
    if not log_folder.exists():
        return []
    sessions = []
    for d in log_folder.iterdir():
        if d.is_dir() and d.joinpath("__session__").exists():
            sessions.append(d.name)
    return sorted(sessions, reverse=True)


def parse_event(tag: str, content: Any, timestamp: datetime) -> Event | None:
    loop_id = extract_loop_id(tag)
    stage = extract_stage(tag)

    # Scenario
    if tag == "scenario":
        return Event(type="scenario", timestamp=timestamp, tag=tag, title="Scenario", content=content)

    # Settings
    if "SETTINGS" in tag:
        name = tag.replace("_SETTINGS", "").replace("SETTINGS", "")
        return Event(type="settings", timestamp=timestamp, tag=tag, title=f"Settings: {name}", content=content)

    # Hypothesis
    if "hypothesis" in tag:
        return Event(
            type="hypothesis",
            timestamp=timestamp,
            tag=tag,
            title="Hypothesis",
            content=content,
            loop_id=loop_id,
            stage="proposal",
        )

    # LLM Call
    if "debug_llm" in tag:
        if isinstance(content, dict) and ("user" in content or "system" in content):
            duration = None
            if content.get("start") and content.get("end"):
                duration = (content["end"] - content["start"]).total_seconds()
            return Event(
                type="llm_call",
                timestamp=timestamp,
                tag=tag,
                title="LLM Call",
                content=content,
                loop_id=loop_id,
                stage=stage,
                duration=duration,
            )

    # Template
    if "debug_tpl" in tag:
        if isinstance(content, dict) and "uri" in content:
            uri = content.get("uri", "")
            tpl_name = uri.split(":")[-1] if ":" in uri else uri
            return Event(
                type="template",
                timestamp=timestamp,
                tag=tag,
                title=f"Template: {tpl_name}",
                content=content,
                loop_id=loop_id,
                stage=stage,
            )

    # Experiment/Coder result
    if "coder result" in tag or "experiment generation" in tag:
        return Event(
            type="experiment",
            timestamp=timestamp,
            tag=tag,
            title="Experiment",
            content=content,
            loop_id=loop_id,
            stage=stage or "coding",
        )

    # Code
    if "evolving code" in tag or "code" in tag.lower():
        return Event(
            type="code",
            timestamp=timestamp,
            tag=tag,
            title="Code",
            content=content,
            loop_id=loop_id,
            stage=stage or "coding",
        )

    # Docker run
    if "docker_run" in tag:
        exit_code = content.get("exit_code") if isinstance(content, dict) else None
        success = exit_code == 0 if exit_code is not None else None
        return Event(
            type="docker_exec",
            timestamp=timestamp,
            tag=tag,
            title=f"Docker Run {'✓' if success else '✗' if success is False else ''}",
            content=content,
            loop_id=loop_id,
            stage="running",
            success=success,
        )

    # Benchmark result
    if "benchmark" in tag.lower():
        return Event(
            type="feedback",
            timestamp=timestamp,
            tag=tag,
            title="Benchmark Result",
            content=content,
            loop_id=loop_id,
            stage="running",
        )

    # Feedback
    if "feedback" in tag:
        decision = getattr(content, "decision", None)
        return Event(
            type="feedback",
            timestamp=timestamp,
            tag=tag,
            title=f"Feedback: {'Accept' if decision else 'Reject'}",
            content=content,
            loop_id=loop_id,
            stage="feedback",
            success=decision,
        )

    # Token cost
    if "token_cost" in tag:
        if isinstance(content, dict):
            total = content.get("total_tokens", 0)
            return Event(
                type="token",
                timestamp=timestamp,
                tag=tag,
                title=f"Token: {total}",
                content=content,
                loop_id=loop_id,
                stage=stage,
            )

    # Time info
    if "time_info" in tag:
        return Event(
            type="time",
            timestamp=timestamp,
            tag=tag,
            title="Time Info",
            content=content,
            loop_id=loop_id,
            stage=stage,
        )

    return None


@st.cache_data(ttl=300, hash_funcs={Path: str})
def load_session(log_path: Path) -> Session:
    """Load events into hierarchical session structure"""
    session = Session()
    
    # 手动遍历 pkl 文件，跳过无法加载的
    events = []
    pkl_files = sorted(log_path.rglob("*.pkl"))
    for pkl_file in pkl_files:
        if pkl_file.name == "debug_llm.pkl":
            continue
        try:
            with open(pkl_file, "rb") as f:
                content = pickle.load(f)
            timestamp = datetime.strptime(pkl_file.stem, "%Y-%m-%d_%H-%M-%S-%f")
            # 正确解析 tag：Loop_5/running/debug_tpl/2957404/xxx.pkl -> Loop_5.running.debug_tpl
            tag = ".".join(pkl_file.relative_to(log_path).as_posix().replace("/", ".").split(".")[:-3])
            event = parse_event(tag, content, timestamp)
            if event:
                events.append(event)
        except (ModuleNotFoundError, ImportError, pickle.UnpicklingError, ValueError):
            # 跳过无法加载的文件（不同 Python 版本或格式错误）
            continue

    events.sort(key=lambda e: e.timestamp)

    for event in events:
        if event.loop_id is None:
            session.init_events.append(event)
            continue

        if event.loop_id not in session.loops:
            session.loops[event.loop_id] = Loop(loop_id=event.loop_id)
        loop = session.loops[event.loop_id]

        if event.stage == "proposal":
            loop.proposal.append(event)
        elif event.stage == "coding":
            loop.coding.append(event)
        elif event.stage == "running":
            loop.running.append(event)
        elif event.stage == "feedback":
            loop.feedback.append(event)
        else:
            loop.proposal.append(event)

    return session


def get_summary(session: Session) -> dict:
    """Get summary statistics"""
    llm_calls = []
    docker_execs = []

    for e in session.init_events:
        if e.type == "llm_call":
            llm_calls.append(e)
        elif e.type == "docker_exec":
            docker_execs.append(e)

    for loop in session.loops.values():
        for e in loop.proposal + loop.coding + loop.running + loop.feedback:
            if e.type == "llm_call":
                llm_calls.append(e)
            elif e.type == "docker_exec":
                docker_execs.append(e)

    return {
        "loop_count": len(session.loops),
        "llm_call_count": len(llm_calls),
        "llm_total_time": sum(e.duration or 0 for e in llm_calls),
        "docker_success": sum(1 for e in docker_execs if e.success is True),
        "docker_fail": sum(1 for e in docker_execs if e.success is False),
    }

