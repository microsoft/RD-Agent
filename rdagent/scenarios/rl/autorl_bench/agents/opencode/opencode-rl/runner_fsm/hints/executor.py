from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._util import _is_truthy, _parse_json_str_list
from .normalizer import (
    normalize_hint_command,
    _first_command_line,
    _extract_cli_flag_value,
    _extract_cli_flag_value_any,
    _DOCKER_LINE_RE,
    _DOTTED_MODULE_RE,
    _ENV_ASSIGN_RE,
)
from .probing import (
    _probe_hint_command,
    _docker_available,
    _hint_runtime_compatible,
    _is_remote_openai_hint,
)
from .scoring import (
    _parse_pytest_counts,
    _extract_score_from_text,
    _extract_score_from_json_obj,
)
from .python_env import (
    _infer_uv_python_candidates,
    _try_uv_venv_retry,
)


@dataclass(frozen=True)
class HintAttempt:
    raw: str
    sanitized: str
    rc: int
    seconds: float
    timed_out: bool
    stdout_tail: str
    stderr_tail: str
    skip_reason: str | None = None


@dataclass(frozen=True)
class HintProbe:
    raw: str
    sanitized: str
    ok: bool | None
    reason: str
    priority: int


def _tail(text: str, n: int) -> str:
    t = str(text or "")
    if len(t) <= n:
        return t
    return t[-n:]


def _run_in_bash(
    cmd: str,
    *,
    workdir: Path,
    env: dict[str, str],
    timeout_seconds: float,
    login_shell: bool,
) -> tuple[int, str, str, bool]:
    """Run a command via bash and return (rc, stdout, stderr, timed_out)."""
    try:
        bash_args = ["bash", "-lc", cmd] if login_shell else ["bash", "-c", cmd]
        res = subprocess.run(
            bash_args, check=False, capture_output=True, text=True,
            timeout=float(timeout_seconds), cwd=str(workdir), env=env,
        )
        return int(res.returncode), res.stdout or "", res.stderr or "", False
    except subprocess.TimeoutExpired as e:
        out_t = getattr(e, "stdout", "") or ""
        err_t = getattr(e, "stderr", "") or ""
        if isinstance(out_t, bytes):
            out_t = out_t.decode("utf-8", errors="replace")
        if isinstance(err_t, bytes):
            err_t = err_t.decode("utf-8", errors="replace")
        return 124, str(out_t), str(err_t), True


def _candidate_metrics_paths(cmd: str, *, repo: Path, workdir: Path | None = None) -> list[Path]:
    """Infer likely output paths for evaluation metrics from a hint command."""
    repo = Path(repo).resolve()
    base = Path(workdir).resolve() if workdir is not None else repo
    out: list[Path] = []
    out_dir = _extract_cli_flag_value_any(
        cmd,
        [
            "--output-dir", "--output_dir", "--out-dir", "--out_dir",
            "--outdir", "--results-dir", "--results_dir",
        ],
    )
    if out_dir:
        p = Path(out_dir.strip())
        if not p.is_absolute():
            p = (base / p).resolve()
        for name in ("metrics.json", "results.json", "summary.json"):
            out.append((p / name).resolve())
    out_path = _extract_cli_flag_value_any(cmd, ["--metrics", "--metrics-path", "--metrics_path"])
    if out_path:
        p = Path(out_path.strip())
        if not p.is_absolute():
            p = (base / p).resolve()
        out.append(p.resolve())
    return out


def _matched_anchors(text: str, *, anchors: list[str]) -> list[str]:
    if not anchors:
        return []
    low = str(text or "").lower()
    seen: set[str] = set()
    out: list[str] = []
    for raw in anchors:
        a = str(raw or "").strip()
        if not a:
            continue
        if a in seen:
            continue
        if a.lower() in low:
            seen.add(a)
            out.append(a)
    return out


def run_hints(
    *,
    repo: Path,
    max_attempts: int = 3,
    timeout_seconds: int = 600,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    env2 = dict(env or os.environ)

    raw_hints = _parse_json_str_list(env2.get("OPENCODE_FSM_HINTS_JSON"))
    if not raw_hints:
        hints_file: Path | None = None
        artifacts_root = (repo / ".opencode_fsm" / "artifacts").resolve()
        if artifacts_root.exists():
            candidates = list(artifacts_root.glob("*/scaffold/scaffold_command_hints.txt"))
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                hints_file = candidates[0]
        if hints_file is not None:
            try:
                text = hints_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = ""
            raw_hints = []
            for raw in text.splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                raw_hints.append(line)

    anchors = _parse_json_str_list(env2.get("OPENCODE_FSM_HINT_ANCHORS_JSON"))
    used_anchors: list[str] = []

    kind = (env2.get("OPENCODE_LLM_KIND") or "").strip().lower()
    prefer_offline = _is_truthy(env2.get("OPENCODE_FSM_PREFER_OFFLINE_HINTS"))
    login_shell = _is_truthy(env2.get("OPENCODE_FSM_HINT_LOGIN_SHELL"))
    strict_compat = _is_truthy(env2.get("OPENCODE_FSM_HINT_STRICT_COMPAT", "1"))
    require_real_score = _is_truthy(env2.get("OPENCODE_FSM_REQUIRE_REAL_SCORE"))
    auto_uv_venv = _is_truthy(env2.get("OPENCODE_FSM_HINT_AUTO_UV", env2.get("OPENCODE_FSM_HINT_AUTO_UV_PY311", "1")))
    artifacts_dir_s = str(env2.get("OPENCODE_FSM_ARTIFACTS_DIR") or "").strip()
    artifacts_dir: Path | None = None
    if artifacts_dir_s:
        try:
            p = Path(artifacts_dir_s).expanduser()
            artifacts_dir = p.resolve() if p.is_absolute() else (repo / p).resolve()
        except Exception:
            artifacts_dir = None

    uv_py_candidates = _infer_uv_python_candidates(repo, env=env2)
    uv_py_env_default = str(env2.get("OPENCODE_FSM_HINT_UV_PYTHON") or env2.get("UV_PYTHON") or "").strip()
    if not uv_py_env_default and sys.version_info >= (3, 13) and uv_py_candidates:
        uv_py_env_default = uv_py_candidates[0]

    uv_hint_env: dict[str, str] | None = None
    uv_hint_env_py: str = ""

    ranked_hints: list[tuple[int, str]] = []
    for raw in raw_hints:
        s = str(raw or "").lower()
        p = 0
        if "pytest" in s:
            p += 50
        if "evaluate" in s or "evaluation" in s:
            p += 20
        if "benchmark" in s:
            p += 10
        has_eval = ("pytest" in s) or ("evaluate" in s) or ("evaluation" in s) or ("benchmark" in s)
        if "docker build" in s and not has_eval:
            p -= 5
        if ("pip install" in s or "poetry install" in s or "conda install" in s) and not has_eval:
            p -= 10
        if prefer_offline:
            if " --samples" in f" {s} ":
                p += 15
            if " --backend openai" in s or " backend openai" in s:
                p -= 30
        else:
            if " --backend openai" in s or " backend openai" in s:
                p += 20
            if "openai" in s and kind == "remote":
                p += 5
        if s.startswith("docker "):
            p += 3
        if "vllm" in s and kind == "remote":
            p -= 5
        if "anthropic" in s or "bedrock" in s or "google" in s:
            p -= 5
        if "ollama" in s:
            p -= 3
        ranked_hints.append((p, raw))
    ranked_hints.sort(key=lambda t: t[0], reverse=True)

    attempts: list[HintAttempt] = []
    probes: list[HintProbe] = []
    chosen: str | None = None
    ok = False
    score = 0.0
    reason = ""
    rc0_no_score = False
    docker_status: tuple[bool, str] | None = None
    hint_work_root: Path | None = None

    candidates: list[dict[str, Any]] = []
    probe_timeout = min(max(3, int(timeout_seconds)), 20)
    seen_sanitized: set[str] = set()

    for priority, raw in ranked_hints:
        sanitized, skip_reason = normalize_hint_command(raw, env=env2)
        if skip_reason is not None:
            attempts.append(
                HintAttempt(
                    raw=raw, sanitized=sanitized, rc=0, seconds=0.0,
                    timed_out=False, stdout_tail="", stderr_tail="",
                    skip_reason=skip_reason,
                )
            )
            probes.append(HintProbe(raw=raw, sanitized=sanitized, ok=False, reason=skip_reason, priority=priority))
            continue

        key = str(sanitized or "").strip()
        if key in seen_sanitized:
            attempts.append(
                HintAttempt(
                    raw=raw, sanitized=sanitized, rc=0, seconds=0.0,
                    timed_out=False, stdout_tail="", stderr_tail="",
                    skip_reason="duplicate_sanitized_hint",
                )
            )
            probes.append(
                HintProbe(raw=raw, sanitized=sanitized, ok=False,
                          reason="duplicate_sanitized_hint", priority=priority)
            )
            continue
        seen_sanitized.add(key)

        compat_ok, compat_reason = _hint_runtime_compatible(cmd=sanitized, env=env2, strict_compat=strict_compat)
        if not compat_ok:
            skip = f"incompatible_hint: {compat_reason}"
            attempts.append(
                HintAttempt(
                    raw=raw, sanitized=sanitized, rc=0, seconds=0.0,
                    timed_out=False, stdout_tail="", stderr_tail="",
                    skip_reason=skip,
                )
            )
            probes.append(HintProbe(raw=raw, sanitized=sanitized, ok=False, reason=skip, priority=priority))
            continue

        if _DOCKER_LINE_RE.search(sanitized):
            if docker_status is None:
                docker_status = _docker_available(env=env2)
            if not docker_status[0]:
                skip = f"docker_unavailable: {docker_status[1]}"
                attempts.append(
                    HintAttempt(
                        raw=raw, sanitized=sanitized, rc=0, seconds=0.0,
                        timed_out=False, stdout_tail="", stderr_tail="",
                        skip_reason=skip,
                    )
                )
                probes.append(HintProbe(raw=raw, sanitized=sanitized, ok=False, reason=skip, priority=priority))
                continue

        probe_ok, probe_reason = _probe_hint_command(
            cmd=sanitized, repo=repo, env=env2, timeout_seconds=probe_timeout,
        )
        probes.append(
            HintProbe(
                raw=raw, sanitized=sanitized, ok=probe_ok,
                reason=str(probe_reason or ""), priority=priority,
            )
        )
        candidates.append(
            {
                "raw": raw, "sanitized": sanitized, "priority": int(priority),
                "probe_ok": probe_ok, "probe_reason": str(probe_reason or ""),
            }
        )

    candidates.sort(
        key=lambda x: (
            (2 if x.get("probe_ok") is True else 1 if x.get("probe_ok") is None else 0),
            int(x.get("priority") or 0),
        ),
        reverse=True,
    )

    picked: set[int] = set()
    ordered: list[dict[str, Any]] = []
    for want in ("pytest", "install", "docker"):
        for i, cand in enumerate(candidates):
            if i in picked:
                continue
            low = str(cand.get("sanitized") or "").lower()
            match = False
            if want == "pytest":
                match = "pytest" in low
            elif want == "install":
                match = ("pip install" in low) or ("poetry install" in low) or ("conda install" in low)
            elif want == "docker":
                match = low.lstrip().startswith("docker ")
            if match:
                ordered.append(cand)
                picked.add(i)
                break
    for i, cand in enumerate(candidates):
        if i in picked:
            continue
        ordered.append(cand)

    executed = 0
    openai_auth_failed = False
    for cand in ordered:
        if executed >= int(max(0, max_attempts)):
            break

        probe_ok = cand.get("probe_ok")
        probe_reason = str(cand.get("probe_reason") or "")
        raw = str(cand.get("raw") or "")
        sanitized = str(cand.get("sanitized") or "")
        first_low = _first_command_line(sanitized).lower()
        looks_openai_codegen = False
        if first_low:
            if ("--samples" not in first_low) and (" -s " not in f" {first_low} "):
                if ("--backend openai" in first_low) or ("--backend=openai" in first_low):
                    if ("--model" in first_low) and ("--dataset" in first_low):
                        if (".evaluate" in first_low) or (".codegen" in first_low):
                            looks_openai_codegen = True

        if openai_auth_failed and _is_remote_openai_hint(sanitized):
            attempts.append(
                HintAttempt(
                    raw=raw, sanitized=sanitized, rc=0, seconds=0.0,
                    timed_out=False, stdout_tail="", stderr_tail="",
                    skip_reason="skipped_after_openai_auth_failure",
                )
            )
            continue

        if probe_ok is False:
            attempts.append(
                HintAttempt(
                    raw=raw, sanitized=sanitized, rc=0, seconds=0.0,
                    timed_out=False, stdout_tail="", stderr_tail="",
                    skip_reason=f"probe_failed: {probe_reason or 'unrunnable'}",
                )
            )
            continue

        attempt_no = int(executed) + 1
        if artifacts_dir is None:
            workdir = repo
        elif _DOCKER_LINE_RE.search(sanitized) or looks_openai_codegen:
            if hint_work_root is None:
                hint_work_root = (artifacts_dir / "hints_workdir").resolve()
                hint_work_root.mkdir(parents=True, exist_ok=True)
            workdir = (hint_work_root / f"attempt_{attempt_no:02d}").resolve()
            workdir.mkdir(parents=True, exist_ok=True)
        else:
            workdir = repo
        metrics_paths = _candidate_metrics_paths(sanitized, repo=repo, workdir=workdir if workdir != repo else None)
        if require_real_score:
            metrics_paths.append((repo / ".opencode_fsm" / "metrics.json").resolve())
        pre_mtimes: dict[Path, float] = {}
        for p in metrics_paths:
            try:
                if p.exists():
                    pre_mtimes[p] = float(p.stat().st_mtime)
            except Exception:
                continue

        t0 = time.monotonic()
        timed_out = False

        env3: dict[str, str] = env2
        if uv_py_env_default:
            env3 = dict(env3)
            env3.setdefault("UV_PYTHON", uv_py_env_default)
        if workdir != repo:
            if env3 is env2:
                env3 = dict(env3)
            repo_s = str(repo)
            pp = str(env3.get("PYTHONPATH") or "")
            parts = [p for p in pp.split(os.pathsep) if p]
            if repo_s not in parts:
                env3["PYTHONPATH"] = pp + (os.pathsep if pp else "") + repo_s
        if workdir != repo and looks_openai_codegen:
            try:
                override_lim = int(str(env3.get("OPENCODE_EVAL_LIMIT") or "").strip() or 0)
            except Exception:
                override_lim = 0
            if override_lim > 0:
                override_dataset = _extract_cli_flag_value(sanitized, "--dataset").strip().lower()
                if override_dataset in ("humaneval", "mbpp"):
                    override_var = (
                        "HUMANEVAL_OVERRIDE_PATH" if override_dataset == "humaneval" else "MBPP_OVERRIDE_PATH"
                    )
                    if not str(env3.get(override_var) or "").strip():
                        override_line = _first_command_line(sanitized)
                        override_module = ""
                        try:
                            override_parts = shlex.split(override_line, posix=True) if override_line else []
                        except Exception:
                            override_parts = []
                        if override_parts:
                            if "-m" in override_parts:
                                try:
                                    override_module = str(override_parts[override_parts.index("-m") + 1] or "").strip()
                                except Exception:
                                    override_module = ""
                            if not override_module:
                                j = 0
                                while j < len(override_parts) and _ENV_ASSIGN_RE.match(str(override_parts[j] or "").strip()):
                                    j += 1
                                if j < len(override_parts):
                                    override_first = str(override_parts[j] or "").strip()
                                    override_mod_cand = os.path.basename(override_first)
                                    if _DOTTED_MODULE_RE.fullmatch(override_mod_cand):
                                        override_module = override_mod_cand
                                    elif _DOTTED_MODULE_RE.fullmatch(override_first):
                                        override_module = override_first
                        if override_module and "." in override_module:
                            override_pkg = override_module.split(".", 1)[0].strip()
                            if override_pkg:
                                override_out_path = (
                                    (Path(workdir) / f"{override_dataset}_override_{override_lim}.jsonl").resolve()
                                )

                                reuse_ok = False
                                try:
                                    reuse_ok = (
                                        override_out_path.exists()
                                        and override_out_path.is_file()
                                        and override_out_path.stat().st_size > 0
                                    )
                                except Exception:
                                    reuse_ok = False
                                if reuse_ok:
                                    env3 = dict(env3)
                                    env3[override_var] = str(override_out_path)
                                else:
                                    override_py_exec = str(
                                        env3.get("OPENCODE_FSM_PYTHON") or env3.get("PYTHON") or sys.executable
                                    ).strip() or sys.executable
                                    try:
                                        p2 = Path(override_py_exec).expanduser()
                                        if not p2.is_absolute() and (
                                            "/" in override_py_exec or override_py_exec.startswith((".", "~"))
                                        ):
                                            py_exec_cand = (repo / p2).resolve()
                                            if py_exec_cand.exists():
                                                override_py_exec = str(py_exec_cand)
                                    except Exception:
                                        pass

                                    override_code = r"""
import importlib
import json
import sys
from pathlib import Path

pkg = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
dataset = (sys.argv[2] if len(sys.argv) > 2 else "").strip().lower()
out = Path(sys.argv[3] if len(sys.argv) > 3 else "").expanduser().resolve()
limit = int(sys.argv[4] if len(sys.argv) > 4 else "0")
if not pkg or not out or limit <= 0:
    raise SystemExit(2)

if dataset == "humaneval":
    dm = importlib.import_module(pkg + ".data.humaneval")
    src = dm._ready_human_eval_plus_path()
elif dataset == "mbpp":
    dm = importlib.import_module(pkg + ".data.mbpp")
    src = dm._ready_mbpp_plus_path()
else:
    raise SystemExit(3)

seen = set()
out.parent.mkdir(parents=True, exist_ok=True)
with open(src, "r", encoding="utf-8", errors="replace") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        tid = obj.get("task_id")
        if not isinstance(tid, str) or not tid.strip():
            continue
        if tid in seen:
            continue
        seen.add(tid)
        g.write(s + "\n")
        if len(seen) >= limit:
            break
if len(seen) <= 0:
    raise SystemExit(4)
"""
                                    try:
                                        res = subprocess.run(
                                            [
                                                override_py_exec, "-c", override_code,
                                                override_pkg, override_dataset,
                                                str(override_out_path), str(override_lim),
                                            ],
                                            check=False, capture_output=True, text=True,
                                            timeout=60, cwd=str(repo), env=env3,
                                        )
                                        if int(res.returncode) == 0 and override_out_path.exists():
                                            env3 = dict(env3)
                                            env3[override_var] = str(override_out_path)
                                    except Exception:
                                        pass

        rc, out, err, this_timed_out = _run_in_bash(
            sanitized, workdir=workdir, env=env3,
            timeout_seconds=timeout_seconds, login_shell=login_shell,
        )
        if this_timed_out:
            timed_out = True

        if rc != 0 and not this_timed_out and auto_uv_venv:
            tail_text = (_tail(out, 20000) + "\n" + _tail(err, 20000)).lower()
            _incompat_needles = (
                ("greenlet", ("cframe", "_pycframe", "failed to build")),
                ("failed building wheel for",),
                ("could not build wheels for",),
                ("subprocess-exited-with-error", ("error:", "failed")),
                ("failed to build installable wheels", ("pyproject.toml", "greenlet")),
            )
            looks_incompat = False
            for needle_group in _incompat_needles:
                if isinstance(needle_group, tuple) and len(needle_group) == 2 and isinstance(needle_group[1], tuple):
                    if needle_group[0] in tail_text and any(n in tail_text for n in needle_group[1]):
                        looks_incompat = True
                        break
                elif isinstance(needle_group, tuple):
                    if all(n in tail_text for n in needle_group):
                        looks_incompat = True
                        break

            if looks_incompat and (
                sys.version_info >= (3, 13) or "cp313" in tail_text
                or "python 3.13" in tail_text or "py3.13" in tail_text
            ):
                retry_result = _try_uv_venv_retry(
                    sanitized, repo=repo, env2=env2,
                    uv_py_candidates=uv_py_candidates,
                    uv_hint_env=uv_hint_env, uv_hint_env_py=uv_hint_env_py,
                    auto_uv_venv=auto_uv_venv, login_shell=login_shell,
                    timeout_seconds=timeout_seconds, workdir=workdir,
                )
                if retry_result is not None:
                    rc, out, err, retry_to, uv_hint_env, uv_hint_env_py = retry_result
                    if retry_to:
                        timed_out = True

        executed += 1
        dt = time.monotonic() - t0
        attempts.append(
            HintAttempt(
                raw=raw, sanitized=sanitized, rc=rc, seconds=float(dt),
                timed_out=timed_out, stdout_tail=_tail(out, 4000),
                stderr_tail=_tail(err, 4000), skip_reason=None,
            )
        )

        low_cmd = sanitized.lower()

        if rc == 0 and require_real_score:
            extracted: float | None = None
            source = ""

            if "pytest" in low_cmd:
                counts = _parse_pytest_counts(_tail(out, 20000) + "\n" + _tail(err, 20000))
                if counts is not None:
                    passed, failed, errors = counts
                    total = max(1, passed + failed + errors)
                    extracted = float(passed) / float(total)
                    source = f"pytest_counts: passed={passed} failed={failed} errors={errors}"

            if extracted is None:
                for p in metrics_paths:
                    try:
                        if not p.exists():
                            continue
                        mt = float(p.stat().st_mtime)
                        if p in pre_mtimes and mt <= pre_mtimes[p] + 1e-6:
                            continue
                    except Exception:
                        continue
                    try:
                        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                    except Exception as e:
                        val, src = None, f"metrics_json_parse_failed:{e}"
                    else:
                        val, src = _extract_score_from_json_obj(data)
                    if val is not None:
                        extracted = float(val)
                        source = f"file:{p.name}:{src}"
                        break

            if extracted is None:
                val, src = _extract_score_from_text(_tail(out, 20000) + "\n" + _tail(err, 20000))
                if val is not None:
                    extracted = float(val)
                    source = src

            if extracted is not None:
                chosen = sanitized
                ok = True
                score = float(extracted)
                reason = str(source or "ok")
                used_anchors = _matched_anchors(sanitized, anchors=anchors)
                break

            rc0_no_score = True
            continue

        if rc == 0:
            chosen = sanitized
            ok = True
            score = 1.0
            reason = ""
            used_anchors = _matched_anchors(sanitized, anchors=anchors)
            break

        if "pytest" in low_cmd:
            counts = _parse_pytest_counts(_tail(out, 20000) + "\n" + _tail(err, 20000))
            if counts is not None:
                passed, failed, errors = counts
                total = max(1, passed + failed + errors)
                chosen = sanitized
                ok = True
                score = float(passed) / float(total)
                reason = f"pytest_nonzero_exit: passed={passed} failed={failed} errors={errors}"
                used_anchors = _matched_anchors(sanitized, anchors=anchors)
                break

        if _is_remote_openai_hint(sanitized):
            low = (_tail(out, 12000) + "\n" + _tail(err, 12000)).lower()
            needles = (
                "invalid_api_key", "incorrect api key provided",
                "authenticationerror", "error code: 401",
                "status': 401", "status: 401",
            )
            if any(n in low for n in needles):
                openai_auth_failed = True

    if not ok:
        if not raw_hints:
            reason = "no_hints"
        elif require_real_score and rc0_no_score:
            reason = "all_hints_no_real_score"
        elif openai_auth_failed:
            reason = "all_hints_auth_failed_or_unrunnable"
        elif candidates and not any((c.get("probe_ok") is not False) for c in candidates):
            reason = "all_hints_unrunnable"
        elif any(a.skip_reason == "unresolved_brackets" for a in attempts):
            reason = "all_hints_unresolved_or_failed"
        else:
            reason = "all_hints_failed"

    return {
        "ok": bool(ok),
        "score": float(score) if ok else 0.0,
        "chosen_command": chosen,
        "used_anchors": used_anchors,
        "executed_attempts": int(executed),
        "probes": [
            {
                "raw": p.raw, "sanitized": p.sanitized, "ok": p.ok,
                "reason": p.reason, "priority": p.priority,
            }
            for p in probes
        ],
        "attempts": [
            {
                "raw": a.raw, "sanitized": a.sanitized, "rc": a.rc,
                "seconds": a.seconds, "timed_out": a.timed_out,
                "stdout_tail": a.stdout_tail, "stderr_tail": a.stderr_tail,
                "skip_reason": a.skip_reason,
            }
            for a in attempts
        ],
        "reason": reason,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run doc-derived hint commands with best-effort sanitization.")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--timeout-seconds", type=int, default=600)
    args = ap.parse_args()

    repo_root = Path(os.environ.get("OPENCODE_FSM_REPO_ROOT") or ".").resolve()
    res = run_hints(repo=repo_root, max_attempts=int(args.max_attempts), timeout_seconds=int(args.timeout_seconds))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
