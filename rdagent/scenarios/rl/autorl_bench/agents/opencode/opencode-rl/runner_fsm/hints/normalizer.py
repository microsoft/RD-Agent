from __future__ import annotations

import json
import re
import shlex
import shutil
import sys
from pathlib import Path

from ..utils.security import cmd_allowed
from .._util import _is_truthy

_FLAG_VALUE_RE = re.compile(r"(?P<flag>--[A-Za-z0-9_.-]+)\s+(?P<val>(?:\"[^\"]*\"|'[^']*'|\S+))")
_BRACKET_GROUP_RE = re.compile(r"\[([^\]]+)\]")
_ANGLE_GROUP_RE = re.compile(r"<[^>]+>")
_GHA_EXPR_RE = re.compile(r"\$\{\{\s*([^}]+)\s*\}\}")
_PIPE_TO_BASH_RE = re.compile(r"(?i)\b(?:curl|wget)\b[^\n]*\|[^\n]*\bbash\b")
_DOTTED_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.]+$")
_DOCKER_LINE_RE = re.compile(r"(?im)^\s*docker\s+")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SHELL_BUILTINS = {
    ":", ".", "alias", "bg", "break", "builtin", "cd", "command", "continue",
    "dirs", "echo", "eval", "exec", "exit", "export", "false", "fg", "hash",
    "help", "history", "jobs", "kill", "local", "popd", "printf", "pushd",
    "pwd", "read", "readonly", "return", "set", "shift", "source", "test",
    "times", "trap", "true", "type", "typeset", "ulimit", "umask", "unalias",
    "unset", "wait",
}


def _replace_flag_value(cmd: str, *, flag: str, new_value: str) -> str:
    flag = flag.strip()
    if not flag:
        return cmd
    if not new_value:
        return cmd
    v = new_value
    if any(ch.isspace() for ch in v):
        v = json.dumps(v)
    return _FLAG_VALUE_RE.sub(
        lambda m: (m.group(0) if m.group("flag") != flag else f"{flag} {v}"),
        cmd,
    )


def _first_command_line(cmd: str) -> str:
    for raw in str(cmd or "").splitlines():
        line = raw.strip()
        if line:
            return line
    return ""


def _extract_cli_flag_value(cmd: str, flag: str) -> str:
    line = _first_command_line(cmd)
    if not line:
        return ""
    try:
        parts = shlex.split(line, posix=True)
    except Exception:
        return ""
    i = 0
    while i < len(parts):
        tok = str(parts[i] or "")
        if tok == flag and i + 1 < len(parts):
            return str(parts[i + 1] or "").strip()
        if tok.startswith(flag + "="):
            return str(tok.split("=", 1)[1] or "").strip()
        i += 1
    return ""


def _extract_cli_flag_value_any(cmd: str, flags: list[str]) -> str:
    for flag in list(flags or []):
        v = _extract_cli_flag_value(cmd, str(flag))
        if v:
            return v
    return ""


def normalize_hint_command(cmd: str, *, env: dict[str, str]) -> tuple[str, str | None]:
    """Normalize a doc-derived command hint into something runnable.

    Returns (sanitized_cmd, skip_reason). If skip_reason is not None, callers should skip it.
    """
    s = str(cmd or "").strip()
    if not s:
        return "", "empty"

    cleaned: list[str] = []
    for raw in s.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("> "):
            line = line[2:].lstrip()
        if line.startswith("$") and len(line) >= 2 and line[1].isspace():
            line = line[2:].lstrip()
        if line.startswith(">>> "):
            line = line[4:].lstrip()
        if line.startswith("... "):
            line = line[4:].lstrip()
        cleaned.append(line)
    s = "\n".join(cleaned).strip()
    if not s:
        return "", "empty_after_sanitize"

    s2 = _BRACKET_GROUP_RE.sub(
        lambda m: (
            str(m.group(1) or "").split("|", 1)[0].strip()
            if "|" in str(m.group(1) or "")
            else m.group(0)
        ),
        s,
    )
    s2 = _ANGLE_GROUP_RE.sub("", s2)

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    s2 = _GHA_EXPR_RE.sub(
        lambda m: (
            py_ver
            if (
                "matrix.python-version" in (inner := str(m.group(1) or "").strip().lower())
                or "matrix.python_version" in inner
                or "python-version" in inner
            )
            else ""
        ),
        s2,
    )

    model = (env.get("OPENCODE_LLM_MODEL") or env.get("OPENAI_MODEL") or "").strip()
    base_url = (env.get("OPENAI_API_BASE") or env.get("OPENAI_BASE_URL") or "").strip()
    if model:
        s2 = _replace_flag_value(s2, flag="--model", new_value=model)
    if base_url:
        s2 = _replace_flag_value(s2, flag="--base-url", new_value=base_url)
        s2 = _replace_flag_value(s2, flag="--base_url", new_value=base_url)

    s2 = re.sub(r"[ \t]+", " ", s2)
    lines: list[str] = []
    for raw in s2.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    s2 = "\n".join(lines).strip()
    if not s2:
        return "", "empty_after_sanitize"

    py = (env.get("OPENCODE_FSM_PYTHON") or env.get("PYTHON") or "python3").strip() or "python3"
    repo_root = str(env.get("OPENCODE_FSM_REPO_ROOT") or "").strip()
    if repo_root and ("/" in py or py.startswith((".", "~"))):
        try:
            p = Path(py).expanduser()
            if not p.is_absolute():
                cand = (Path(repo_root).expanduser().resolve() / p).resolve()
                if cand.exists():
                    py = str(cand)
        except Exception:
            pass

    rewritten: list[str] = []
    for line in s2.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parts = shlex.split(line, posix=True)
        except Exception:
            rewritten.append(line)
            continue
        if not parts:
            rewritten.append(line)
            continue
        first = str(parts[0] or "").strip()
        if not first or "/" in first or first.startswith((".", "~")):
            rewritten.append(line)
            continue
        if _DOTTED_MODULE_RE.fullmatch(first) and shutil.which(first) is None:
            rest = " ".join(shlex.quote(str(p)) for p in parts[1:])
            base = f"{shlex.quote(py)} -m {shlex.quote(first)}"
            rewritten.append(f"{base} {rest}".strip() if rest else base)
        else:
            rewritten.append(line)
    s2 = "\n".join(rewritten).strip()

    py_is_path = ("/" in py) or py.startswith((".", "~"))
    py_first = shlex.split(shlex.quote(py))[0]
    rewritten_tools: list[str] = []
    for raw in s2.splitlines():
        line = raw.strip()
        if not line:
            continue
        if py_is_path:
            try:
                parts = shlex.split(line, posix=True)
            except Exception:
                rewritten_tools.append(line)
                continue
            if not parts:
                rewritten_tools.append(line)
                continue

            prefix: list[str] = []
            i = 0
            while i < len(parts) and _ENV_ASSIGN_RE.match(str(parts[i] or "").strip()):
                prefix.append(str(parts[i] or ""))
                i += 1
            if i < len(parts) and str(parts[i] or "") == "env":
                prefix.append("env")
                i += 1
                while i < len(parts) and _ENV_ASSIGN_RE.match(str(parts[i] or "").strip()):
                    prefix.append(str(parts[i] or ""))
                    i += 1
            if i >= len(parts):
                rewritten_tools.append(line)
                continue

            cmd0 = str(parts[i] or "")
            rest = [str(x or "") for x in parts[i + 1 :]]

            tok = cmd0.strip()
            is_py = tok in ("python", "python3") or bool(re.fullmatch(r"python\\d+(?:\\.\\d+)?", tok))
            is_pip = tok in ("pip", "pip3") or bool(re.fullmatch(r"pip\\d+(?:\\.\\d+)?", tok))

            if cmd0 != py and is_py:
                cmd0 = py
            elif is_pip:
                cmd0 = py
                rest = ["-m", "pip"] + rest
            elif cmd0 == "pytest":
                cmd0 = py
                rest = ["-m", "pytest"] + rest

            line = " ".join(shlex.quote(x) for x in (prefix + [cmd0] + rest)).strip()
        rewritten_tools.append(line)
    s2 = "\n".join(rewritten_tools).strip()

    fire_aliases = {
        "--base-url": "--base_url",
        "--n-samples": "--n_samples",
        "--id-range": "--id_range",
        "--i-just-wanna-run": "--i_just_wanna_run",
        "--test-details": "--test_details",
        "--base-only": "--base_only",
        "--output-file": "--output_file",
        "--min-time-limit": "--min_time_limit",
        "--gt-time-limit-factor": "--gt_time_limit_factor",
    }
    bounded: list[str] = []
    for line in s2.splitlines():
        looks_fire = False
        try:
            parts = shlex.split(line, posix=True)
        except Exception:
            parts = []
        if parts:
            i = 0
            while i < len(parts) and _ENV_ASSIGN_RE.match(str(parts[i] or "").strip()):
                i += 1
            if i < len(parts):
                first = str(parts[i] or "").strip()
                if first:
                    if _DOTTED_MODULE_RE.fullmatch(first):
                        looks_fire = True
                    elif first in ("python", "python3", py_first):
                        if i + 2 < len(parts) and str(parts[i + 1]) == "-m":
                            mod = str(parts[i + 2] or "").strip()
                            if _DOTTED_MODULE_RE.fullmatch(mod):
                                looks_fire = True
        if looks_fire:
            for old, new in fire_aliases.items():
                line = line.replace(old, new)

        low = line.lower()
        if ("--samples" not in low) and (" -s " not in f" {low} "):
            if ("--backend openai" in low) or ("--backend=openai" in low):
                if ("--model" in low) and ("--dataset" in low) and ((".evaluate" in low) or (".codegen" in low)):
                    parts2 = line.split()
                    has_n_samples = any(p.startswith("--n_samples") for p in parts2)
                    if not has_n_samples:
                        line = (line + " --n_samples 1").strip()

        bounded.append(line)
    s2 = "\n".join(bounded)

    strip_pytest_n = _is_truthy(env.get("OPENCODE_FSM_HINT_STRIP_PYTEST_N", "1"))
    if strip_pytest_n:
        stripped: list[str] = []
        for line in s2.splitlines():
            try:
                parts = shlex.split(line, posix=True)
            except Exception:
                stripped.append(line)
                continue
            if not parts:
                stripped.append(line)
                continue
            if "pytest" not in parts:
                stripped.append(line)
                continue
            out: list[str] = []
            i = 0
            while i < len(parts):
                tok = str(parts[i] or "")
                if tok == "-n":
                    if i + 1 < len(parts) and not str(parts[i + 1] or "").startswith("-"):
                        i += 2
                    else:
                        i += 1
                    continue
                if tok.startswith("-n="):
                    i += 1
                    continue
                if tok == "--dist":
                    if i + 1 < len(parts) and not str(parts[i + 1] or "").startswith("-"):
                        i += 2
                    else:
                        i += 1
                    continue
                if tok.startswith("--dist="):
                    i += 1
                    continue
                out.append(tok)
                i += 1
            stripped.append(" ".join(shlex.quote(x) for x in out))
        s2 = "\n".join(stripped)

    tmp = re.sub(r"\[\s*\d+\s*,\s*\d+\s*\]", "", s2)
    if "[" in tmp and "]" in tmp:
        return s2, "unresolved_brackets"
    if "<" in s2 and ">" in s2:
        return s2, "unresolved_angle_placeholders"

    if _PIPE_TO_BASH_RE.search(s2):
        return s2, "blocked_pipe_to_bash"

    allowed, reason = cmd_allowed(s2, pipeline=None)
    if not allowed:
        return s2, reason or "blocked_by_policy"
    return s2, None
