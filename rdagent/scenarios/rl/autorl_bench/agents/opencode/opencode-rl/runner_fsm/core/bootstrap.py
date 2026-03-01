from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._util import resolve_workdir
from .pipeline_spec import PipelineSpec
from ..utils.security import cmd_allowed, looks_interactive, safe_env
from ..utils.subprocess import read_text_if_exists, run_cmd_capture, write_cmd_artifacts, write_json, write_text
from ..dtypes import CmdResult, StageResult

@dataclass(frozen=True)
class BootstrapSpec:

    version: int = 1
    cmds: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None
    timeout_seconds: int | None = None
    retries: int = 0

@dataclass(frozen=True)
class BootstrapLoadResult:
    """Structured bootstrap parse output with non-fatal normalization warnings."""

    spec: BootstrapSpec
    raw: str
    warnings: list[str] = field(default_factory=list)

_VAR_BRACE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_VAR_BARE_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_DOLLAR_PLACEHOLDER = "\x00DOLLAR\x00"

def _normalize_bootstrap_applied_env_paths(repo: Path, applied_env: dict[str, str]) -> dict[str, str]:
    """Normalize common path-like bootstrap env values to be robust across workdirs.

    Why this exists:
    - Many scaffolded `bootstrap.yml` files set values like:
      - `PATH: ".opencode_fsm/venv/bin:$PATH"`
      - `OPENCODE_FSM_PYTHON: ".opencode_fsm/venv/bin/python"`
    - Some stages (notably `runner.hints_exec`) may execute commands from an isolated
      workdir under `$OPENCODE_FSM_ARTIFACTS_DIR`, where relative PATH segments would
      no longer resolve to the repo's venv.
    """
    root = Path(repo).resolve()
    out = dict(applied_env or {})

    py = str(out.get("OPENCODE_FSM_PYTHON") or "").strip()
    if py:
        try:
            p = Path(py)
            if not p.is_absolute():
                # Do not use `.resolve()` here: venv interpreters are often symlinks to the
                # base interpreter, and resolving the symlink breaks venv isolation.
                out["OPENCODE_FSM_PYTHON"] = str((root / p).absolute())
        except Exception:
            pass

    raw_path = str(out.get("PATH") or "").strip()
    if raw_path:
        parts = raw_path.split(os.pathsep)
        new_parts: list[str] = []
        changed = False
        for item in parts:
            seg = str(item or "")
            if seg == "":
                # Preserve empty segments (meaning: current directory).
                new_parts.append("")
                continue
            if "$" in seg:
                new_parts.append(seg)
                continue
            try:
                p2 = Path(seg)
            except Exception:
                new_parts.append(seg)
                continue
            if p2.is_absolute():
                new_parts.append(seg)
                continue
            new_parts.append(str((root / p2).absolute()))
            changed = True
        if changed:
            out["PATH"] = os.pathsep.join(new_parts)

    return out

def _coerce_cmds(raw: Any, *, field_name: str, warnings: list[str]) -> list[str]:
    out: list[str] = []

    if raw is None:
        return out

    if isinstance(raw, str):
        s = raw.strip()
        if s:
            warnings.append(f"bootstrap.{field_name}_coerced_from_string")
            return [s]
        return out

    if isinstance(raw, dict):
        cmd = raw.get("cmd")
        if cmd is None:
            cmd = raw.get("run")
        if cmd is None:
            cmd = raw.get("command")
        if isinstance(cmd, str) and cmd.strip():
            warnings.append(f"bootstrap.{field_name}_mapping_coerced_to_single_cmd")
            return [cmd.strip()]
        raise ValueError(
            f"bootstrap.{field_name} mapping must contain one of keys: cmd, run, command "
            "(with non-empty string value)"
        )

    if not isinstance(raw, list):
        raise ValueError(f"bootstrap.{field_name} must be a list of non-empty strings")

    for i, item in enumerate(raw):
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
            continue
        if isinstance(item, dict):
            cmd = item.get("cmd")
            if cmd is None:
                cmd = item.get("run")
            if cmd is None:
                cmd = item.get("command")
            if isinstance(cmd, str) and cmd.strip():
                out.append(cmd.strip())
                warnings.append(f"bootstrap.{field_name}[{i}]_mapping_coerced_via_run")
                continue
        raise ValueError(
            f"bootstrap.{field_name}[{i}] must be a non-empty string "
            "or mapping with cmd/run/command"
        )
    return out

def load_bootstrap_spec_with_diagnostics(path: Path) -> BootstrapLoadResult:
    """Load bootstrap.yml and tolerate common scaffold formatting variants.

    This keeps parsing strict enough for safety while handling typical agent outputs
    (e.g. `boot:` wrapper, `steps` alias, mapping-style step items).
    """
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: PyYAML. Install with `pip install PyYAML`.") from e

    raw = read_text_if_exists(path).strip()
    if not raw:
        raise ValueError(f"bootstrap file is empty: {path}")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("bootstrap.yml must be a YAML mapping (dict) at the top level")

    warnings: list[str] = []
    obj = dict(data or {})
    boot = obj.get("boot")
    if isinstance(boot, dict):
        top_level_keys = {"cmds", "steps", "env", "workdir", "cwd", "timeout_seconds", "timeout", "retries", "retry"}
        has_top_level_spec = any(k in obj for k in top_level_keys)
        if not has_top_level_spec:
            merged = dict(boot)
            for k, v in obj.items():
                if k == "boot":
                    continue
                merged[k] = v
            obj = merged
            warnings.append("bootstrap.boot_mapping_unwrapped")
        else:
            warnings.append("bootstrap.boot_mapping_ignored_due_to_top_level_fields")

    version = int(obj.get("version") or 1)
    if version != 1:
        raise ValueError(f"unsupported bootstrap version: {version}")

    cmds_src = obj.get("cmds")
    if cmds_src is None and obj.get("steps") is not None:
        cmds_src = obj.get("steps")
        warnings.append("bootstrap.steps_alias_used")
    cmds = _coerce_cmds(cmds_src, field_name="cmds", warnings=warnings)

    env = obj.get("env") or {}
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise ValueError("bootstrap.env must be a mapping")
    env_out: dict[str, str] = {}
    for k, v in env.items():
        if k is None:
            continue
        ks = str(k).strip()
        if not ks:
            continue
        env_out[ks] = "" if v is None else str(v)

    workdir_raw = obj.get("workdir")
    if workdir_raw is None and obj.get("cwd") is not None:
        workdir_raw = obj.get("cwd")
        warnings.append("bootstrap.cwd_alias_used")
    workdir = str(workdir_raw).strip() if workdir_raw else None

    timeout_raw = obj.get("timeout_seconds")
    if timeout_raw is None and obj.get("timeout") is not None:
        timeout_raw = obj.get("timeout")
        warnings.append("bootstrap.timeout_alias_used")
    timeout_seconds = None
    if timeout_raw is not None:
        s = str(timeout_raw).strip()
        if s:
            try:
                timeout_seconds = int(s)
            except Exception as e:
                raise ValueError("bootstrap.timeout_seconds must be an integer") from e
            if timeout_seconds < 1:
                raise ValueError("bootstrap.timeout_seconds must be >= 1")

    retries_raw = obj.get("retries")
    if retries_raw is None and obj.get("retry") is not None:
        retries_raw = obj.get("retry")
        warnings.append("bootstrap.retry_alias_used")
    retries = None
    if retries_raw is not None:
        s = str(retries_raw).strip()
        if s:
            try:
                retries = int(s)
            except Exception as e:
                raise ValueError("bootstrap.retries must be an integer") from e
            if retries < 0:
                raise ValueError("bootstrap.retries must be >= 0")
    if retries is None:
        retries = 0

    spec = BootstrapSpec(
        version=version,
        cmds=[c.strip() for c in cmds if c.strip()],
        env=env_out,
        workdir=workdir,
        timeout_seconds=timeout_seconds,
        retries=int(retries),
    )
    return BootstrapLoadResult(spec=spec, raw=raw, warnings=warnings)

def load_bootstrap_spec(path: Path) -> tuple[BootstrapSpec, str]:
    loaded = load_bootstrap_spec_with_diagnostics(path)
    return loaded.spec, loaded.raw

def run_bootstrap(
    repo: Path,
    *,
    bootstrap_path: Path,
    pipeline: PipelineSpec | None,
    unattended: str,
    artifacts_dir: Path,
) -> tuple[StageResult, dict[str, str]]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        loaded = load_bootstrap_spec_with_diagnostics(bootstrap_path)
        spec, raw = loaded.spec, loaded.raw
        if loaded.warnings:
            write_json(artifacts_dir / "bootstrap_parse_warnings.json", {"warnings": list(loaded.warnings)})
    except Exception as e:
        res = CmdResult(cmd=f"parse {bootstrap_path}", rc=2, stdout="", stderr=str(e), timed_out=False)
        write_cmd_artifacts(artifacts_dir, "bootstrap_parse", res)
        write_json(artifacts_dir / "bootstrap_summary.json", {"ok": False, "failed_index": 0, "total_results": 1})
        return StageResult(ok=False, results=[res], failed_index=0), {}

    write_text(artifacts_dir / "bootstrap.yml", raw)

    # Seed commonly-used variables for env expansion.
    env_base = dict(os.environ)
    env_base["OPENCODE_FSM_REPO_ROOT"] = str(repo.resolve())

    env_for_cmds = dict(env_base)
    applied_env: dict[str, str] = {}
    for k, v in (spec.env or {}).items():
        key = str(k or "").strip()
        if not key:
            continue
        s = str(v or "")
        s = s.replace("$$", _DOLLAR_PLACEHOLDER)
        s = _VAR_BRACE_RE.sub(lambda m: str(env_for_cmds.get(m.group(1)) or ""), s)
        s = _VAR_BARE_RE.sub(lambda m: str(env_for_cmds.get(m.group(1)) or ""), s)
        value = s.replace(_DOLLAR_PLACEHOLDER, "$")
        env_for_cmds[key] = value
        applied_env[key] = value

    applied_env = _normalize_bootstrap_applied_env_paths(repo, applied_env)
    env_for_cmds.update(dict(applied_env))
    env_for_cmds = safe_env(env_for_cmds, {}, unattended=unattended)
    env_for_cmds["OPENCODE_FSM_STAGE"] = "bootstrap"
    env_for_cmds["OPENCODE_FSM_ARTIFACTS_DIR"] = str(artifacts_dir.resolve())
    env_for_cmds["OPENCODE_FSM_REPO_ROOT"] = str(repo.resolve())
    redacted: dict[str, str] = {}
    for k, v in (applied_env or {}).items():
        ku = str(k or "").upper()
        if any(x in ku for x in ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASS", "PWD")):
            redacted[str(k)] = "***redacted***"
        else:
            redacted[str(k)] = "" if v is None else str(v)
    write_json(artifacts_dir / "bootstrap_env.json", redacted)

    try:
        workdir = resolve_workdir(repo, spec.workdir)
    except Exception as e:
        res = CmdResult(cmd=f"resolve_workdir {spec.workdir}", rc=2, stdout="", stderr=str(e), timed_out=False)
        write_cmd_artifacts(artifacts_dir, "bootstrap_workdir_error", res)
        write_json(artifacts_dir / "bootstrap_summary.json", {"ok": False, "failed_index": 0, "total_results": 1})
        return StageResult(ok=False, results=[res], failed_index=0), applied_env

    results: list[CmdResult] = []
    failed_index: int | None = None

    # No cmds is valid: env-only bootstrap.
    for cmd_idx, raw_cmd in enumerate(spec.cmds, start=1):
        cmd = raw_cmd.strip()
        if not cmd:
            continue

        if unattended == "strict" and looks_interactive(cmd):
            res = CmdResult(
                cmd=cmd,
                rc=126,
                stdout="",
                stderr="likely_interactive_command_disallowed_in_strict_mode",
                timed_out=False,
            )
            results.append(res)
            failed_index = len(results) - 1
            write_cmd_artifacts(artifacts_dir, f"bootstrap_cmd{cmd_idx:02d}_try01", res)
            break

        allowed, reason = cmd_allowed(cmd, pipeline=pipeline)
        if not allowed:
            res = CmdResult(cmd=cmd, rc=126, stdout="", stderr=reason or "blocked", timed_out=False)
            results.append(res)
            failed_index = len(results) - 1
            write_cmd_artifacts(artifacts_dir, f"bootstrap_cmd{cmd_idx:02d}_try01", res)
            break

        eff_timeout = spec.timeout_seconds
        if pipeline and pipeline.security_max_cmd_seconds:
            eff_timeout = (
                int(pipeline.security_max_cmd_seconds)
                if eff_timeout is None
                else min(int(eff_timeout), int(pipeline.security_max_cmd_seconds))
            )

        for attempt in range(1, int(spec.retries) + 2):
            res = run_cmd_capture(cmd, workdir, timeout_seconds=eff_timeout, env=env_for_cmds, interactive=False)
            results.append(res)
            write_cmd_artifacts(artifacts_dir, f"bootstrap_cmd{cmd_idx:02d}_try{attempt:02d}", res)
            if res.rc == 0:
                break

        if results and results[-1].rc != 0:
            failed_index = len(results) - 1
            break

    ok = failed_index is None
    write_json(artifacts_dir / "bootstrap_summary.json", {"ok": ok, "failed_index": failed_index, "total_results": len(results)})

    return StageResult(ok=ok, results=results, failed_index=failed_index), applied_env
