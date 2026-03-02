from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

@dataclass(frozen=True)
class CmdResult:
    cmd: str
    rc: int
    stdout: str
    stderr: str
    timed_out: bool = False

@dataclass(frozen=True)
class TurnEvent:
    """单轮 agent 交互的事件，用于流式输出。"""
    turn: int
    assistant_text: str = ""
    calls: list[Any] = ()      # list[ToolCall]
    results: list[Any] = ()    # list[ToolResult]
    finished: bool = False

@dataclass(frozen=True)
class AgentResult:
    assistant_text: str
    raw: Any | None = None
    tool_trace: list[dict[str, Any]] | None = None

class AgentClient(Protocol):
    def run(self, text: str, *, fsm_state: str, iter_idx: int, purpose: str, on_turn: Callable[[TurnEvent], None] | None = None) -> AgentResult:
        ...
    def close(self) -> None:
        ...
