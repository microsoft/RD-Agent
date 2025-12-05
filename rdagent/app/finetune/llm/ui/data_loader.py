"""
FT UI Data Loader
Load pkl logs and convert to hierarchical timeline structure
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from rdagent.log.storage import FileStorage

EventType = Literal["scenario", "llm_call", "template", "experiment", "code", "docker_exec", "feedback", "token", "time", "settings", "hypothesis", "dataset_selection"]


@dataclass
class Event:
    """Timeline event"""
    type: EventType
    timestamp: datetime
    tag: str
    title: str
    content: Any
    loop_id: int | None = None
    evo_id: int | None = None
    stage: str = ""
    duration: float | None = None
    success: bool | None = None

    @property
    def time_str(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")


@dataclass
class EvoLoop:
    """Evolution loop containing events"""
    evo_id: int
    events: list[Event] = field(default_factory=list)
    success: bool | None = None


@dataclass
class Loop:
    """Main loop containing stages"""
    loop_id: int
    exp_gen: list[Event] = field(default_factory=list)
    coding: dict[int, EvoLoop] = field(default_factory=dict)  # evo_id -> EvoLoop
    runner: list[Event] = field(default_factory=list)
    feedback: list[Event] = field(default_factory=list)


@dataclass
class Session:
    """Session containing init events and loops"""
    init_events: list[Event] = field(default_factory=list)
    loops: dict[int, Loop] = field(default_factory=dict)  # loop_id -> Loop


def extract_loop_id(tag: str) -> int | None:
    match = re.search(r"Loop_(\d+)", tag)
    return int(match.group(1)) if match else None


def extract_evo_id(tag: str) -> int | None:
    match = re.search(r"evo_loop_(\d+)", tag)
    return int(match.group(1)) if match else None


def extract_stage(tag: str) -> str:
    if "direct_exp_gen" in tag:
        return "exp_gen"
    if "coding" in tag:
        return "coding"
    if "running" in tag:  # Note: tag uses "running", not "runner"
        return "runner"
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
    evo_id = extract_evo_id(tag)
    stage = extract_stage(tag)

    # Scenario
    if tag == "scenario":
        model = getattr(content, "base_model", "Unknown")
        return Event(type="scenario", timestamp=timestamp, tag=tag,
                     title=f"Scenario: {model}", content=content)

    # Dataset selection
    if "dataset_selection" in tag:
        selected = content.get("selected_datasets", []) if isinstance(content, dict) else []
        total = content.get("total_datasets", 0) if isinstance(content, dict) else 0
        return Event(type="dataset_selection", timestamp=timestamp, tag=tag,
                     title=f"Dataset Selection: {len(selected)}/{total}", content=content)

    # Settings
    if "SETTINGS" in tag:
        name = tag.replace("_SETTINGS", "").replace("SETTINGS", "")
        return Event(type="settings", timestamp=timestamp, tag=tag,
                     title=f"Settings: {name}", content=content)

    # Hypothesis
    if tag == "hypothesis" or (loop_id is not None and "hypothesis" in tag):
        return Event(type="hypothesis", timestamp=timestamp, tag=tag,
                     title="Hypothesis", content=content,
                     loop_id=loop_id, stage="exp_gen")

    # LLM Call
    if "debug_llm" in tag:
        if isinstance(content, dict) and "system" in content:
            duration = None
            if content.get("start") and content.get("end"):
                duration = (content["end"] - content["start"]).total_seconds()
            return Event(type="llm_call", timestamp=timestamp, tag=tag,
                         title="LLM Call", content=content,
                         loop_id=loop_id, evo_id=evo_id, stage=stage, duration=duration)

    # Template
    if "debug_tpl" in tag:
        if isinstance(content, dict) and "uri" in content:
            uri = content.get("uri", "")
            tpl_name = uri.split(":")[-1] if ":" in uri else uri
            return Event(type="template", timestamp=timestamp, tag=tag,
                         title=f"Template: {tpl_name}", content=content,
                         loop_id=loop_id, evo_id=evo_id, stage=stage)

    # Experiment generation
    if "experiment generation" in tag:
        task_count = len(content) if isinstance(content, list) else 1
        return Event(type="experiment", timestamp=timestamp, tag=tag,
                     title=f"Experiment ({task_count} task)", content=content,
                     loop_id=loop_id, stage=stage)

    # Evolving code
    if "evolving code" in tag:
        file_count = 0
        if isinstance(content, list):
            for ws in content:
                if hasattr(ws, "file_dict"):
                    file_count += len(ws.file_dict)
        return Event(type="code", timestamp=timestamp, tag=tag,
                     title=f"Code ({file_count} files)", content=content,
                     loop_id=loop_id, evo_id=evo_id, stage=stage or "coding")

    # Docker execution (individual evaluator feedback)
    if "docker_exec." in tag:
        class_name = tag.split("docker_exec.")[-1].split(".")[0]
        # Map class names to display names
        display_names = {
            "FTDataEvaluator": "Data Processing",
            "FTCoderEvaluator": "Micro-batch Test",
            "FTRunnerEvaluator": "Full Train",
        }
        evaluator_name = display_names.get(class_name, class_name)
        # Single feedback object (new format)
        success = getattr(content, "final_decision", None)
        title = f"Docker ({evaluator_name}) {'✓' if success else '✗' if success is False else '?'}"
        return Event(type="docker_exec", timestamp=timestamp, tag=tag,
                     title=title, content=content, loop_id=loop_id, evo_id=evo_id,
                     stage=stage or "coding", success=success)

    # Evolving feedback
    if "evolving feedback" in tag:
        success = None
        if hasattr(content, "feedback_list") and content.feedback_list:
            fb = content.feedback_list[0]
            success = getattr(fb, "final_decision", None)
        title = "Running(Full Train)" if stage == "runner" else f"Docker {'✓' if success else '✗' if success is False else '?'}"
        return Event(type="docker_exec", timestamp=timestamp, tag=tag,
                     title=title, content=content, loop_id=loop_id, evo_id=evo_id,
                     stage=stage or "coding", success=success)

    # Final feedback
    if "feedback.feedback" in tag or (tag.endswith(".feedback") and "evo_loop" not in tag):
        decision = getattr(content, "decision", None)
        return Event(type="feedback", timestamp=timestamp, tag=tag,
                     title=f"Feedback: {'Accept' if decision else 'Reject'}",
                     content=content, loop_id=loop_id, stage="feedback", success=decision)

    # Runner result
    if "runner result" in tag:
        return Event(type="docker_exec", timestamp=timestamp, tag=tag,
                     title="Full Train", content=content,
                     loop_id=loop_id, stage="runner")

    # Token cost
    if "token_cost" in tag:
        if isinstance(content, dict):
            total = content.get("total_tokens", 0)
            return Event(type="token", timestamp=timestamp, tag=tag,
                         title=f"Token: {total}", content=content,
                         loop_id=loop_id, evo_id=evo_id, stage=stage)

    # Time info
    if "time_info" in tag:
        return Event(type="time", timestamp=timestamp, tag=tag,
                     title="Time Info", content=content,
                     loop_id=loop_id, stage=stage)

    return None


def load_ft_session(log_path: Path) -> Session:
    """Load events into hierarchical session structure"""
    session = Session()
    storage = FileStorage(log_path)

    events = []
    for msg in storage.iter_msg():
        if not msg.tag:
            continue
        event = parse_event(msg.tag, msg.content, msg.timestamp)
        if event:
            events.append(event)

    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp)

    # Organize into hierarchy
    for event in events:
        if event.loop_id is None:
            session.init_events.append(event)
            continue

        # Ensure loop exists
        if event.loop_id not in session.loops:
            session.loops[event.loop_id] = Loop(loop_id=event.loop_id)
        loop = session.loops[event.loop_id]

        # Place event in appropriate stage
        if event.stage == "exp_gen":
            loop.exp_gen.append(event)
        elif event.stage == "coding":
            if event.evo_id is not None:
                if event.evo_id not in loop.coding:
                    loop.coding[event.evo_id] = EvoLoop(evo_id=event.evo_id)
                evo = loop.coding[event.evo_id]
                evo.events.append(event)
                if event.type == "docker_exec" and event.success is not None:
                    evo.success = event.success
            else:
                # Coding events without evo_id go to evo 0
                if 0 not in loop.coding:
                    loop.coding[0] = EvoLoop(evo_id=0)
                loop.coding[0].events.append(event)
        elif event.stage == "runner":
            loop.runner.append(event)
        elif event.stage == "feedback":
            loop.feedback.append(event)
        else:
            # Unknown stage - put in exp_gen
            loop.exp_gen.append(event)

    return session


def get_summary(session: Session) -> dict:
    """Get summary statistics"""
    llm_calls = []
    docker_execs = []

    # Collect from init
    for e in session.init_events:
        if e.type == "llm_call":
            llm_calls.append(e)
        elif e.type == "docker_exec":
            docker_execs.append(e)

    # Collect from loops
    for loop in session.loops.values():
        for e in loop.exp_gen + loop.runner + loop.feedback:
            if e.type == "llm_call":
                llm_calls.append(e)
            elif e.type == "docker_exec":
                docker_execs.append(e)
        for evo in loop.coding.values():
            for e in evo.events:
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
