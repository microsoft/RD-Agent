from __future__ import annotations

import configparser
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from .._util import _parse_json_str_list

_PY_MAJOR_MINOR_RE = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)")


def _as_major_minor(raw: str | None) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    m = _PY_MAJOR_MINOR_RE.search(s)
    if not m:
        return ""
    try:
        major = int(m.group("major"))
        minor = int(m.group("minor"))
    except Exception:
        return ""
    if major <= 0 or minor < 0:
        return ""
    return f"{major}.{minor}"


def _infer_repo_python_pin(repo: Path) -> str:
    """Infer a repo's preferred Python major.minor from common version pin files."""
    repo = Path(repo).resolve()
    for rel in (".python-version", "runtime.txt"):
        p = (repo / rel).resolve()
        try:
            if not p.exists() or not p.is_file():
                continue
            for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                mm = _as_major_minor(line)
                if mm:
                    return mm
                break
        except Exception:
            continue

    p = (repo / ".tool-versions").resolve()
    try:
        if p.exists() and p.is_file():
            for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if not line.startswith("python"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    mm = _as_major_minor(parts[1])
                    if mm:
                        return mm
    except Exception:
        pass
    return ""


def _infer_repo_requires_python(repo: Path) -> str:
    """Infer a repo's `requires-python` style spec (best-effort)."""
    repo = Path(repo).resolve()
    pyproject = (repo / "pyproject.toml").resolve()
    if pyproject.exists() and pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict):
                proj = data.get("project")
                if isinstance(proj, dict):
                    rp = proj.get("requires-python")
                    if isinstance(rp, str) and rp.strip():
                        return rp.strip()
                tool = data.get("tool")
                if isinstance(tool, dict):
                    poetry = tool.get("poetry")
                    if isinstance(poetry, dict):
                        deps = poetry.get("dependencies")
                        if isinstance(deps, dict):
                            py = deps.get("python")
                            if isinstance(py, str) and py.strip():
                                return py.strip()
        except Exception:
            pass

    setup_cfg = (repo / "setup.cfg").resolve()
    if setup_cfg.exists() and setup_cfg.is_file():
        try:
            cp = configparser.ConfigParser()
            cp.read(setup_cfg, encoding="utf-8")
            if cp.has_option("options", "python_requires"):
                v = str(cp.get("options", "python_requires") or "").strip()
                if v:
                    return v
        except Exception:
            pass

    setup_py = (repo / "setup.py").resolve()
    if setup_py.exists() and setup_py.is_file():
        try:
            text = setup_py.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"(?i)python_requires\\s*=\\s*['\\\"]([^'\\\"]+)['\\\"]", text)
            if m:
                v = str(m.group(1) or "").strip()
                if v:
                    return v
        except Exception:
            pass
    return ""


def _best_python_minor_from_spec(spec: str, *, candidates: list[str]) -> str:
    """Pick the highest major.minor in candidates that satisfies a python spec string."""
    s = str(spec or "").strip()
    if not s:
        return ""
    try:
        from packaging.specifiers import SpecifierSet  # type: ignore
        from packaging.version import Version  # type: ignore

        ss = SpecifierSet(s)
        for mm in candidates:
            try:
                if Version(f"{mm}.0") in ss:
                    return mm
            except Exception:
                continue
    except Exception:
        pass

    for mm in candidates:
        if mm in s:
            return mm
    m = re.search(r"<\\s*(\\d+)\\.(\\d+)", s)
    if m:
        try:
            major = int(m.group(1))
            minor = int(m.group(2))
        except Exception:
            major = 0
            minor = 0
        if major > 0:
            want = f"{major}.{max(0, minor - 1)}"
            if want in candidates:
                return want
    return ""


def _infer_uv_python_candidates(repo: Path, *, env: dict[str, str]) -> list[str]:
    """Infer a list of uv `--python` requests to try (most preferred first)."""
    env2 = dict(env or {})
    out: list[str] = []

    raw_candidates = _parse_json_str_list(env2.get("OPENCODE_FSM_HINT_UV_PYTHON_CANDIDATES_JSON"))
    if raw_candidates:
        out.extend([c.strip() for c in raw_candidates if isinstance(c, str) and c.strip()])
    else:
        single = str(env2.get("OPENCODE_FSM_HINT_UV_PYTHON") or env2.get("UV_PYTHON") or "").strip()
        if single:
            out.append(single)

    if not out:
        pinned = _infer_repo_python_pin(repo)
        if pinned:
            out.append(pinned)

    if not out:
        spec = _infer_repo_requires_python(repo)
        if spec:
            prefer = ["3.12", "3.11", "3.10", "3.9", "3.8"]
            picked = _best_python_minor_from_spec(spec, candidates=prefer)
            if picked:
                out.append(picked)
            else:
                mm = _as_major_minor(spec)
                if mm:
                    out.append(mm)

    if not out and sys.version_info >= (3, 13):
        out.extend(["3.12", "3.11"])

    seen: set[str] = set()
    cleaned: list[str] = []
    for v in out:
        s = str(v or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    return cleaned


def _try_uv_venv_retry(
    sanitized: str,
    *,
    repo: Path,
    env2: dict[str, str],
    uv_py_candidates: list[str],
    uv_hint_env: dict[str, str] | None,
    uv_hint_env_py: str,
    auto_uv_venv: bool,
    login_shell: bool,
    timeout_seconds: float,
    workdir: Path,
) -> tuple[int, str, str, bool, dict[str, str] | None, str] | None:
    """Attempt to retry a failed command inside a uv-managed venv.

    Returns (rc, stdout, stderr, timed_out, updated_uv_env, uv_py) on retry,
    or None if retry was not applicable.
    """
    from .executor import _run_in_bash

    if uv_hint_env is not None:
        rc2, out2, err2, to2 = _run_in_bash(
            sanitized, workdir=workdir, env=uv_hint_env,
            timeout_seconds=timeout_seconds, login_shell=login_shell,
        )
        extra = f"cached python={uv_hint_env_py}" if uv_hint_env_py else "cached"
        return rc2, out2, f"(retry_uv_venv: {extra})\n{err2}", to2, uv_hint_env, uv_hint_env_py

    if not auto_uv_venv:
        return None
    if shutil.which("uv") is None:
        return None
    if not uv_py_candidates:
        return None

    raw_venv_dir = str(env2.get("OPENCODE_FSM_HINT_UV_VENV_DIR") or "").strip()
    uv_try = [uv_py_candidates[0]] if raw_venv_dir else list(uv_py_candidates)
    last_err = ""
    for py_req in uv_try:
        try:
            m = re.match(r"^\s*(\d+)\.(\d+)\s*$", py_req)
            tag = f"py{m.group(1)}{m.group(2)}" if m else "py"
        except Exception:
            tag = "py"

        if raw_venv_dir:
            venv_dir = Path(raw_venv_dir).expanduser()
            if not venv_dir.is_absolute():
                venv_dir = (repo / venv_dir).resolve()
        else:
            venv_dir = (repo / ".opencode_fsm" / f"venv_hints_{tag}").resolve()
        py_bin = (venv_dir / "bin" / "python").absolute()
        try:
            venv_dir.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            uv_res = subprocess.run(
                ["uv", "venv", "--allow-existing", "--seed", "pip", "--python", py_req, str(venv_dir)],
                check=False, capture_output=True, text=True, timeout=600,
                cwd=str(repo), env=env2,
            )
        except Exception as e:
            last_err = f"uv_venv_failed:{e}"
            continue
        if int(getattr(uv_res, "returncode", 1)) != 0:
            t = _tail(str(getattr(uv_res, "stderr", "") or "") + "\n" + str(getattr(uv_res, "stdout", "") or ""), 2500)
            last_err = f"uv_venv_failed_rc={getattr(uv_res, 'returncode', None)}:{t}"
            continue

        envx = dict(env2)
        old_path = str(envx.get("PATH") or "")
        envx["PATH"] = str((venv_dir / "bin").absolute()) + (os.pathsep + old_path if old_path else "")
        envx["VIRTUAL_ENV"] = str(venv_dir.absolute())
        envx["OPENCODE_FSM_PYTHON"] = str(py_bin)
        envx["PYTHON"] = str(py_bin)
        envx.setdefault("UV_PYTHON", str(py_req).strip())

        rc2, out2, err2, to2 = _run_in_bash(
            sanitized, workdir=workdir, env=envx,
            timeout_seconds=timeout_seconds, login_shell=login_shell,
        )
        extra = f"ok python={py_req}"
        return rc2, out2, f"(retry_uv_venv: {extra})\n{err2}", to2, envx, py_req.strip()

    return None


def _tail(text: str, n: int) -> str:
    t = str(text or "")
    if len(t) <= n:
        return t
    return t[-n:]
