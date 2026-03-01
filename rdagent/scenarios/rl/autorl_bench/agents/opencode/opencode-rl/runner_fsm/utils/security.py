from __future__ import annotations

import re
import shlex
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.pipeline_spec import PipelineSpec

_HARD_DENY_PATTERNS: tuple[str, ...] = (
    r"(^|[;&|]\s*)\s*rm\s+-rf\s+/\s*($|[;&|])",
    r"(^|[;&|]\s*)\s*rm\s+-rf\s+/\*\s*($|[;&|])",
    r"(^|[;&|]\s*)\s*rm\s+-rf\s+~\s*($|[;&|])",
    r"(^|[;&|]\s*)\s*rm\s+-rf\s+\$HOME\s*($|[;&|])",
    r"(^|[;&|]\s*)\s*:\(\)\s*\{\s*:\|\:\s*&\s*\}\s*;\s*:\s*($|[;&|])",  # fork bomb
)

_SAFE_DEFAULT_DENY_PATTERNS: tuple[str, ...] = (
    r"\bsudo\b",
    r"\bbrew\s+uninstall\b",
    r"\bdocker\s+system\s+prune\b",
    r"\bdocker\s+volume\s+prune\b",
    r"\bmkfs\b",
    r"\bdd\b",
    r"\bshutdown\b",
    r"\breboot\b",
)

_SCRIPT_HARD_DENY_PATTERNS: tuple[str, ...] = (
    r"\brm\s+-rf\s+/\s*($|[;&|])",
    r"\brm\s+-rf\s+/\*\s*($|[;&|])",
    r"\brm\s+-rf\s+~\s*($|[;&|])",
    r"\brm\s+-rf\s+\$HOME\s*($|[;&|])",
    # Runner evidence lives here; deleting it destroys debuggability and audit trails.
    r"\brm\s+-rf\s+\.opencode_fsm/artifacts\b",
    r":\(\)\s*\{\s*:\|\:\s*&\s*\}\s*;\s*:\s*",
)

_SCRIPT_SAFE_DENY_PATTERNS: tuple[str, ...] = (
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bdd\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bdocker\s+system\s+prune\b",
    r"\bdocker\s+volume\s+prune\b",
    r"\bcurl\b[^\n]*\|[^\n]*\bbash\b",
    r"\bwget\b[^\n]*\|[^\n]*\bbash\b",
)

def compile_patterns(patterns: list[str] | tuple[str, ...]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for raw in patterns:
        p = str(raw)
        if not p.strip():
            continue
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error:
            compiled.append(re.compile(re.escape(p), re.IGNORECASE))
    return compiled

def matches_any(patterns: list[re.Pattern[str]], text: str) -> str | None:
    for p in patterns:
        if p.search(text):
            return p.pattern
    return None

def looks_interactive(cmd: str) -> bool:
    s = cmd.strip().lower()
    if not s:
        return False

    # Heuristics to avoid hanging in strict unattended runs.
    if s.startswith("docker login") and "--password-stdin" not in s and " -p " not in s and " --password " not in s:
        return True
    if " gh auth login" in f" {s}" and "--with-token" not in s:
        return True
    return False

def safe_env(base: dict[str, str], extra: dict[str, str], *, unattended: str) -> dict[str, str]:
    env = dict(base)
    env.update({k: str(v) for k, v in extra.items()})
    if unattended == "strict":
        env.setdefault("CI", "1")
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
    return env

def cmd_allowed(cmd: str, *, pipeline: PipelineSpec | None) -> tuple[bool, str | None]:
    cmd = cmd.strip()
    if not cmd:
        return False, "empty_command"

    hard_deny = compile_patterns(_HARD_DENY_PATTERNS)
    hit = matches_any(hard_deny, cmd)
    if hit:
        return False, f"blocked_by_hard_deny: {hit}"

    # If no pipeline is provided, default to safe mode deny patterns.
    if pipeline is None:
        deny = compile_patterns(_SAFE_DEFAULT_DENY_PATTERNS)
        hit = matches_any(deny, cmd)
        if hit:
            return False, f"blocked_by_default_safe_deny: {hit}"
        return True, None

    mode = (pipeline.security_mode or "safe").strip().lower()
    if mode not in ("safe", "system"):
        return False, f"invalid_security_mode: {mode}"

    deny_patterns = list(pipeline.security_denylist or [])
    if mode == "safe":
        deny_patterns.extend(list(_SAFE_DEFAULT_DENY_PATTERNS))
    deny = compile_patterns(deny_patterns)
    hit = matches_any(deny, cmd)
    if hit:
        return False, f"blocked_by_denylist: {hit}"

    allow_patterns = list(pipeline.security_allowlist or [])
    if allow_patterns:
        allow = compile_patterns(allow_patterns)
        if matches_any(allow, cmd) is None:
            return False, "blocked_by_allowlist"

    return True, None

def audit_bash_script(
    cmd: str,
    *,
    repo: Path,
    workdir: Path,
    pipeline: PipelineSpec | None,
) -> tuple[bool, str | None]:
    s = str(cmd or "").strip()
    if not s:
        return True, None

    try:
        parts = shlex.split(s, posix=True)
    except Exception:
        return True, None

    if not parts:
        return True, None

    exe = str(parts[0] or "").strip().lower()
    if exe not in ("bash", "sh"):
        return True, None
    if len(parts) < 2:
        return True, None

    script_token = str(parts[1] or "").strip()
    if not script_token or script_token.startswith("-"):
        return True, None

    repo = Path(repo).resolve()
    workdir = Path(workdir).resolve()
    script_path = Path(script_token)
    if not script_path.is_absolute():
        script_path = (workdir / script_path).resolve()
    else:
        script_path = script_path.resolve()

    # Only audit scripts that are within the repo root (avoid surprising behavior for system scripts).
    try:
        script_path.relative_to(repo)
    except Exception:
        return True, None

    if not script_path.exists() or not script_path.is_file():
        return True, None

    try:
        raw = script_path.read_bytes()
    except Exception as e:
        return False, f"blocked_by_script_audit_read_error: {e}"

    # Avoid reading extremely large scripts in full.
    max_bytes = 2_000_000
    if len(raw) > max_bytes:
        return False, f"blocked_by_script_audit_too_large: {len(raw)} bytes"

    text = raw.decode("utf-8", errors="replace")

    mode = (pipeline.security_mode if pipeline else "safe") or "safe"
    mode = str(mode).strip().lower() or "safe"
    deny_patterns: list[str] = list(_SCRIPT_HARD_DENY_PATTERNS)

    if pipeline:
        deny_patterns.extend(list(pipeline.security_denylist or []))
    if mode == "safe" or pipeline is None:
        deny_patterns.extend(list(_SCRIPT_SAFE_DENY_PATTERNS))

    deny = compile_patterns(deny_patterns)
    hit = matches_any(deny, text)
    if hit:
        return False, f"blocked_by_script_denylist: {hit}"
    return True, None
