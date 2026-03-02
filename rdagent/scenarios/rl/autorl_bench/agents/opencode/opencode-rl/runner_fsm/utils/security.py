from __future__ import annotations

import re
from typing import Any

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

def cmd_allowed(cmd: str, *, pipeline: Any | None) -> tuple[bool, str | None]:
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
