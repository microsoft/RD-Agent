from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ..dtypes import CmdResult

STDIO_TAIL_CHARS = 8000
ARTIFACT_TEXT_LIMIT_CHARS = 2_000_000

def tail(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[-n:]

def run_cmd(cmd: str, cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        text=True,
        capture_output=True,
    )
    return p.returncode, tail(p.stdout, STDIO_TAIL_CHARS), tail(p.stderr, STDIO_TAIL_CHARS)

def run_cmd_capture(
    cmd: str,
    cwd: Path,
    *,
    timeout_seconds: int | None = None,
    env: dict[str, str] | None = None,
    interactive: bool = False,
) -> CmdResult:
    try:
        if interactive:
            p = subprocess.run(
                cmd,
                cwd=str(cwd),
                shell=True,
                text=True,
                env=env,
                timeout=timeout_seconds,
            )
            return CmdResult(cmd=cmd, rc=p.returncode, stdout="", stderr="", timed_out=False)

        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            shell=True,
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout_seconds,
        )
        return CmdResult(cmd=cmd, rc=p.returncode, stdout=p.stdout, stderr=p.stderr, timed_out=False)
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="replace")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode(errors="replace")
        return CmdResult(cmd=cmd, rc=124, stdout=stdout, stderr=stderr, timed_out=True)

def limit_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]...\n"

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(limit_text(text, ARTIFACT_TEXT_LIMIT_CHARS), encoding="utf-8", errors="replace")

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")

def write_cmd_artifacts(out_dir: Path, prefix: str, res: CmdResult) -> None:
    write_text(out_dir / f"{prefix}_cmd.txt", res.cmd + "\n")
    write_text(out_dir / f"{prefix}_stdout.txt", res.stdout)
    write_text(out_dir / f"{prefix}_stderr.txt", res.stderr)
    write_json(
        out_dir / f"{prefix}_result.json",
        {"rc": res.rc, "timed_out": res.timed_out},
    )
