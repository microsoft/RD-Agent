from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

from .normalizer import _first_command_line, _extract_cli_flag_value_any, _ENV_ASSIGN_RE, _SHELL_BUILTINS


def _canonical_base_url(url: str | None) -> str:
    s = str(url or "").strip()
    if not s:
        return ""
    s = s.rstrip("/")
    if s.endswith("/v1"):
        return s[: -len("/v1")]
    return s


def _hint_backend(cmd: str) -> str:
    from .normalizer import _extract_cli_flag_value

    backend = _extract_cli_flag_value(cmd, "--backend").strip().lower()
    if backend:
        return backend
    line = _first_command_line(cmd).lower()
    if ".evaluate" in line and "--dataset" in line and "--model" in line:
        return "openai"
    return ""


def _is_remote_openai_hint(cmd: str) -> bool:
    return _hint_backend(cmd) == "openai"


def _hint_runtime_compatible(*, cmd: str, env: dict[str, str], strict_compat: bool) -> tuple[bool, str]:
    if not strict_compat:
        return True, "ok"
    low = _first_command_line(cmd).lower()
    if not low:
        return False, "empty"

    backend = _hint_backend(cmd)
    llm_kind = str(env.get("OPENCODE_LLM_KIND") or "").strip().lower()
    if llm_kind == "remote" and backend and backend != "openai":
        if "--samples" not in low:
            return False, f"backend_mismatch:{backend}"

    runtime_base = _canonical_base_url(env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE"))
    hinted_base = _canonical_base_url(_extract_cli_flag_value_any(cmd, ["--base-url", "--base_url"]))
    if runtime_base and hinted_base and runtime_base != hinted_base:
        return False, "base_url_mismatch"

    return True, "ok"


def _docker_available(*, env: dict[str, str]) -> tuple[bool, str]:
    """Best-effort check for a usable local Docker daemon."""
    if shutil.which("docker") is None:
        return False, "docker_not_found"
    try:
        res = subprocess.run(
            ["docker", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
            env=env,
        )
    except Exception as e:
        return False, f"docker_info_failed: {e}"
    if int(res.returncode) != 0:
        tail = (res.stderr or res.stdout or "").strip()
        if len(tail) > 500:
            tail = tail[-500:]
        return False, tail or f"docker_info_rc={res.returncode}"
    return True, "ok"


def _extract_invoked_command(parts: list[str]) -> tuple[str, list[str]]:
    i = 0
    n = len(parts)
    while i < n:
        tok = str(parts[i] or "").strip()
        if not tok:
            i += 1
            continue
        if _ENV_ASSIGN_RE.match(tok):
            i += 1
            continue
        if tok == "env":
            i += 1
            while i < n and _ENV_ASSIGN_RE.match(str(parts[i] or "").strip()):
                i += 1
            continue
        return tok, parts[i:]
    return "", []


def _probe_hint_command(
    *,
    cmd: str,
    repo: Path,
    env: dict[str, str],
    timeout_seconds: int,
) -> tuple[bool | None, str]:
    """Best-effort non-mutating probe for hint runnability."""
    text = str(cmd or "").strip()
    if not text:
        return False, "empty"

    first_line = ""
    for raw in text.splitlines():
        s = raw.strip()
        if s:
            first_line = s
            break
    if not first_line:
        return False, "empty"

    try:
        parts = shlex.split(first_line, posix=True)
    except Exception:
        return None, "probe_shlex_failed"
    if not parts:
        return False, "empty"

    invoked, tail_parts = _extract_invoked_command(parts)
    if not invoked:
        return None, "probe_no_invoked_command"

    tok = str(invoked).strip()
    if not tok:
        return None, "probe_no_invoked_command"

    if tok in ("bash", "sh", "zsh", "fish"):
        return None, "probe_shell_wrapper"
    if tok in _SHELL_BUILTINS:
        return None, "probe_shell_builtin"
    if "/" in tok or tok.startswith((".", "~")):
        return True, "ok"

    if tok != "docker":
        samples = ""
        i = 0
        while i < len(parts):
            t = str(parts[i] or "").strip()
            if t in ("--samples", "-s") and i + 1 < len(parts):
                samples = str(parts[i + 1] or "").strip()
                break
            if t.startswith("--samples="):
                samples = str(t.split("=", 1)[1] or "").strip()
                break
            i += 1
        if samples and not samples.startswith("-"):
            sp = Path(samples)
            if not sp.is_absolute():
                sp = (repo / sp).resolve()
            try:
                if not sp.exists():
                    return False, f"samples_not_found:{sp}"
            except Exception:
                pass

    py_names = {
        "python",
        "python3",
        Path(str(env.get("OPENCODE_FSM_PYTHON") or "")).name.strip(),
        Path(str(env.get("PYTHON") or "")).name.strip(),
    }
    py_names = {x for x in py_names if x}
    if tok in py_names:
        if len(tail_parts) >= 3 and str(tail_parts[1]) == "-m":
            module = str(tail_parts[2] or "").strip()
            if module:
                probe_py = (
                    str(env.get("OPENCODE_FSM_PYTHON") or "").strip()
                    or str(env.get("PYTHON") or "").strip()
                    or tok
                    or "python3"
                )
                code = (
                    "import importlib.util, sys; "
                    "m = (sys.argv[1] if len(sys.argv) > 1 else '').strip(); "
                    "sys.exit(0 if (m and importlib.util.find_spec(m) is not None) else 3)"
                )
                try:
                    res = subprocess.run(
                        [probe_py, "-c", code, module],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=min(max(2, int(timeout_seconds)), 8),
                        cwd=str(repo),
                        env=env,
                    )
                except Exception as e:
                    return None, f"probe_module_check_failed:{e}"
                if int(res.returncode) != 0:
                    return False, f"module_not_found:{module}"
                return True, "ok"

    if shutil.which(tok) is None:
        return False, f"binary_not_found:{tok}"
    return True, "ok"
