"""Checkpoint 存取：原子写入 / 加载。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .types import PipelineState


def _checkpoint_path(workspace: str) -> Path:
    return Path(workspace) / "checkpoint.json"


def save_checkpoint(state: PipelineState) -> None:
    """原子写入 checkpoint.json。

    先写临时文件再 rename，防止进程中断导致 checkpoint 损坏。
    """
    path = _checkpoint_path(state.workspace)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = json.dumps(state.to_dict(), indent=2, ensure_ascii=False)

    # 原子写入：同目录下创建临时文件，写完后 rename
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".checkpoint_", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp_path, str(path))
    except BaseException:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    try:
        from .ui import console
        console.print(f"  [dim]\\[checkpoint] Saved: iter={state.current_iteration}, phases={len(state.iterations)}[/]")
    except Exception:
        print(f"  [checkpoint] Saved: iter={state.current_iteration}, phases={len(state.iterations)}")


def load_checkpoint(workspace: str) -> PipelineState | None:
    """加载 checkpoint.json，返回 PipelineState 或 None（不存在时）。"""
    path = _checkpoint_path(workspace)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        state = PipelineState.from_dict(data)
        try:
            from .ui import console
            console.print(f"  [dim]\\[checkpoint] Loaded: iter={state.current_iteration}, phases={len(state.iterations)}[/]")
        except Exception:
            pass
        return state
    except Exception as e:
        print(f"  [checkpoint] WARNING: Failed to load {path}: {e}")
        return None
