from __future__ import annotations

import json
import os
import shlex
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .._util import resolve_workdir, _is_truthy, _parse_json_str_list
from .pipeline_spec import PipelineSpec
from ..utils.security import audit_bash_script, cmd_allowed, looks_interactive, safe_env
from ..utils.subprocess import (
    STDIO_TAIL_CHARS,
    read_text_if_exists,
    run_cmd_capture,
    tail,
    write_cmd_artifacts,
    write_json,
    write_text,
)
from ..dtypes import CmdResult, StageResult, VerificationResult

def stage_rc(stage: StageResult | None) -> int | None:
    if stage is None or not stage.results:
        return None
    if stage.failed_index is None:
        return stage.results[-1].rc
    if 0 <= stage.failed_index < len(stage.results):
        return stage.results[stage.failed_index].rc
    return stage.results[-1].rc

def stage_failed_cmd(stage: StageResult | None) -> CmdResult | None:
    if stage is None or not stage.results:
        return None
    if stage.failed_index is not None and 0 <= stage.failed_index < len(stage.results):
        return stage.results[stage.failed_index]
    return stage.results[-1]

def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return None, f"failed_to_read: {e}"
    try:
        data = json.loads(raw)
    except Exception as e:
        return None, f"invalid_json: {e}"
    if not isinstance(data, dict):
        return None, "metrics_json_not_object"
    return data, None

def _dump_kubectl(
    out_dir: Path,
    repo: Path,
    *,
    namespace: str | None,
    label_selector: str | None,
    include_logs: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmds: list[tuple[str, str]] = [
        ("kubectl_get_nodes", "kubectl get nodes -o wide"),
        ("kubectl_get_namespaces", "kubectl get namespaces"),
        ("kubectl_get_pods", "kubectl get pods -A -o wide"),
        ("kubectl_get_all", "kubectl get all -A -o wide"),
        ("kubectl_get_events", "kubectl get events -A --sort-by=.metadata.creationTimestamp"),
    ]
    for prefix, cmd in cmds:
        res = run_cmd_capture(cmd, repo, timeout_seconds=60)
        write_cmd_artifacts(out_dir, prefix, res)

    if include_logs and label_selector:
        if namespace:
            cmd = (
                f"kubectl logs -n {shlex.quote(namespace)} -l {shlex.quote(label_selector)} "
                "--all-containers=true --timestamps --tail=2000"
            )
        else:
            cmd = (
                f"kubectl logs --all-namespaces -l {shlex.quote(label_selector)} "
                "--all-containers=true --timestamps --tail=2000"
            )
        res = run_cmd_capture(cmd, repo, timeout_seconds=120)
        write_cmd_artifacts(out_dir, "kubectl_logs", res)

def _run_stage(
    repo: Path,
    *,
    stage: str,
    cmds: list[str],
    workdir: Path,
    env: dict[str, str],
    timeout_seconds: int | None,
    retries: int,
    interactive: bool,
    unattended: str,
    pipeline: PipelineSpec | None,
    artifacts_dir: Path,
) -> StageResult:
    stage = stage.strip() or "stage"
    env2 = dict(env)
    env2["OPENCODE_FSM_STAGE"] = stage
    env2["OPENCODE_FSM_ARTIFACTS_DIR"] = str(artifacts_dir.resolve())
    env2["OPENCODE_FSM_REPO_ROOT"] = str(repo.resolve())
    results: list[CmdResult] = []
    started = time.monotonic()

    for cmd_idx, raw_cmd in enumerate(cmds, start=1):
        cmd = raw_cmd.strip()
        if not cmd:
            continue

        for attempt in range(1, int(retries) + 2):
            if pipeline and pipeline.security_max_total_seconds:
                elapsed = time.monotonic() - started
                if elapsed > float(pipeline.security_max_total_seconds):
                    res = CmdResult(
                        cmd=cmd, rc=124, stdout="",
                        stderr=f"max_total_seconds_exceeded: {pipeline.security_max_total_seconds}",
                        timed_out=True,
                    )
                    results.append(res)
                    failed_index = len(results) - 1
                    write_cmd_artifacts(artifacts_dir, f"{stage}_cmd{cmd_idx:02d}_try{attempt:02d}", res)
                    write_cmd_artifacts(artifacts_dir, stage, res)
                    write_json(
                        artifacts_dir / f"{stage}_summary.json",
                        {"ok": False, "failed_index": failed_index, "total_results": len(results)},
                    )
                    return StageResult(ok=False, results=results, failed_index=failed_index)

            allowed, reason = cmd_allowed(cmd, pipeline=pipeline)
            if not allowed:
                res = CmdResult(cmd=cmd, rc=126, stdout="", stderr=reason or "blocked", timed_out=False)
            elif unattended == "strict" and looks_interactive(cmd):
                res = CmdResult(
                    cmd=cmd, rc=126, stdout="",
                    stderr="likely_interactive_command_disallowed_in_strict_mode",
                    timed_out=False,
                )
            else:
                ok, audit_reason = audit_bash_script(cmd, repo=repo, workdir=workdir, pipeline=pipeline)
                if not ok:
                    res = CmdResult(cmd=cmd, rc=126, stdout="", stderr=audit_reason or "blocked_by_script_audit", timed_out=False)
                else:
                    eff_timeout = timeout_seconds
                    if pipeline and pipeline.security_max_cmd_seconds:
                        eff_timeout = (
                            int(pipeline.security_max_cmd_seconds)
                            if eff_timeout is None
                            else min(int(eff_timeout), int(pipeline.security_max_cmd_seconds))
                        )
                    res = run_cmd_capture(
                        cmd, workdir,
                        timeout_seconds=eff_timeout,
                        env=env2,
                        interactive=bool(interactive and unattended == "guided"),
                    )

            results.append(res)
            write_cmd_artifacts(artifacts_dir, f"{stage}_cmd{cmd_idx:02d}_try{attempt:02d}", res)

            if res.rc == 0:
                break

        if results and results[-1].rc != 0:
            failed_index = len(results) - 1
            write_cmd_artifacts(artifacts_dir, stage, results[-1])
            write_json(
                artifacts_dir / f"{stage}_summary.json",
                {"ok": False, "failed_index": failed_index, "total_results": len(results)},
            )
            return StageResult(ok=False, results=results, failed_index=failed_index)

    if results:
        write_cmd_artifacts(artifacts_dir, stage, results[-1])
    write_json(
        artifacts_dir / f"{stage}_summary.json",
        {"ok": True, "failed_index": None, "total_results": len(results)},
    )
    return StageResult(ok=True, results=results, failed_index=None)


def _resolve_and_run_stage(
    repo: Path,
    *,
    stage_name: str,
    cmds: list[str],
    workdir_spec: str | None,
    stage_env: dict[str, str],
    env_base: dict[str, str],
    timeout_seconds: int | None,
    retries: int,
    interactive: bool,
    unattended: str,
    pipeline: PipelineSpec | None,
    artifacts_dir: Path,
) -> StageResult:
    """Resolve workdir and run a stage. Returns a failed StageResult on workdir error."""
    env = safe_env(env_base, stage_env, unattended=unattended)
    try:
        workdir = resolve_workdir(repo, workdir_spec)
    except Exception as e:
        err = CmdResult(cmd=f"resolve_workdir {workdir_spec}", rc=2, stdout="", stderr=str(e), timed_out=False)
        write_cmd_artifacts(artifacts_dir, f"{stage_name}_workdir_error", err)
        return StageResult(ok=False, results=[err], failed_index=0)
    return _run_stage(
        repo, stage=stage_name, cmds=cmds, workdir=workdir, env=env,
        timeout_seconds=timeout_seconds, retries=retries,
        interactive=interactive, unattended=unattended,
        pipeline=pipeline, artifacts_dir=artifacts_dir,
    )


def _build_result(
    *,
    ok: bool,
    failed_stage: str | None,
    stage_results: dict[str, StageResult | None],
    metrics_path: str | None = None,
    metrics: dict[str, Any] | None = None,
    metrics_errors: list[str] | None = None,
) -> VerificationResult:
    """Build a VerificationResult from collected stage results."""
    return VerificationResult(
        ok=ok,
        failed_stage=failed_stage,
        auth=stage_results.get("auth"),
        tests=stage_results.get("tests"),
        deploy_setup=stage_results.get("deploy_setup"),
        deploy_health=stage_results.get("deploy_health"),
        rollout=stage_results.get("rollout"),
        evaluation=stage_results.get("evaluation"),
        benchmark=stage_results.get("benchmark"),
        metrics_path=metrics_path,
        metrics=metrics,
        metrics_errors=metrics_errors or [],
    )


def _validate_metrics(
    repo: Path,
    pipeline: PipelineSpec,
    metrics_cfg_path: str | None,
    required_keys: list[str],
    prefix: str,
    artifacts_dir: Path,
) -> tuple[str | None, dict[str, Any] | None, list[str], bool]:
    """Validate metrics file. Returns (metrics_path, metrics_dict, errors, ok)."""
    if not metrics_cfg_path:
        return None, None, [], True

    mpath = Path(metrics_cfg_path).expanduser()
    if not mpath.is_absolute():
        mpath = repo / mpath
    mpath_str = str(mpath)

    if not mpath.exists():
        return mpath_str, None, [f"{prefix}.metrics_file_missing: {mpath}"], False

    write_text(artifacts_dir / f"metrics_{prefix}.json", read_text_if_exists(mpath))

    data, err = _read_json(mpath)
    if err:
        return mpath_str, data, [f"{prefix}.{err}"], False

    missing = [k for k in (required_keys or []) if k not in (data or {})]
    if missing:
        return mpath_str, data, [f"{prefix}.missing_keys: " + ", ".join(missing)], False

    if "ok" in (required_keys or []) and data.get("ok") is not True:
        return mpath_str, data, [f"{prefix}.ok_not_true"], False

    return mpath_str, data, [], True


def run_pipeline_verification(
    repo: Path,
    *,
    pipeline: PipelineSpec | None,
    tests_cmds: list[str],
    artifacts_dir: Path,
    unattended: str = "strict",
) -> VerificationResult:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ok = False
    metrics_path: str | None = None
    metrics: dict[str, Any] | None = None
    metrics_errors: list[str] = []
    stage_results: dict[str, StageResult | None] = {}

    teardown_cmds = list(pipeline.deploy_teardown_cmds or []) if pipeline else []
    teardown_policy = (pipeline.deploy_teardown_policy if pipeline else "never").lower()
    kubectl_dump_enabled = bool(pipeline and pipeline.kubectl_dump_enabled)

    env_base = dict(os.environ)
    runner_root = Path(__file__).resolve().parents[1]
    env_base.setdefault("OPENCODE_FSM_RUNNER_ROOT", str(runner_root))
    env_base.setdefault("OPENCODE_FSM_PYTHON", sys.executable)
    existing_pp = str(env_base.get("PYTHONPATH") or "")
    parts = [p for p in existing_pp.split(os.pathsep) if p]
    root_s = str(runner_root)
    parts = [p for p in parts if p != root_s]
    env_base["PYTHONPATH"] = root_s + (os.pathsep + os.pathsep.join(parts) if parts else "")

    if pipeline is not None:
        max_cmd = None
        raw = env_base.get("OPENCODE_FSM_MAX_CMD_SECONDS")
        if isinstance(raw, str) and raw.strip():
            raw = raw.strip()
        else:
            raw = None
            for m in (
                pipeline.auth_env, pipeline.tests_env, pipeline.deploy_env,
                pipeline.rollout_env, pipeline.evaluation_env, pipeline.benchmark_env,
            ):
                vv = m.get("OPENCODE_FSM_MAX_CMD_SECONDS") if isinstance(m, dict) else None
                if isinstance(vv, str) and vv.strip():
                    raw = vv.strip()
                    break
        if raw is not None:
            try:
                n = int(str(raw).strip())
            except Exception:
                n = None
            max_cmd = n if isinstance(n, int) and n > 0 else None

        max_total = None
        raw = env_base.get("OPENCODE_FSM_MAX_TOTAL_SECONDS")
        if isinstance(raw, str) and raw.strip():
            raw = raw.strip()
        else:
            raw = None
            for m in (
                pipeline.auth_env, pipeline.tests_env, pipeline.deploy_env,
                pipeline.rollout_env, pipeline.evaluation_env, pipeline.benchmark_env,
            ):
                vv = m.get("OPENCODE_FSM_MAX_TOTAL_SECONDS") if isinstance(m, dict) else None
                if isinstance(vv, str) and vv.strip():
                    raw = vv.strip()
                    break
        if raw is not None:
            try:
                n = int(str(raw).strip())
            except Exception:
                n = None
            max_total = n if isinstance(n, int) and n > 0 else None
        if max_cmd is not None or max_total is not None:
            pipeline = replace(
                pipeline,
                security_max_cmd_seconds=max_cmd if max_cmd is not None else pipeline.security_max_cmd_seconds,
                security_max_total_seconds=max_total if max_total is not None else pipeline.security_max_total_seconds,
            )

    try:
        # --- Sequential stages: auth, tests, deploy_setup, deploy_health, rollout ---
        _stages = []
        if pipeline and pipeline.auth_cmds:
            _stages.append(("auth", pipeline.auth_cmds, pipeline.auth_workdir,
                            pipeline.auth_env, pipeline.auth_timeout_seconds,
                            pipeline.auth_retries, bool(pipeline.auth_interactive)))
        # tests always runs (using the provided tests_cmds)
        _stages.append(("tests", tests_cmds, pipeline.tests_workdir if pipeline else None,
                         pipeline.tests_env if pipeline else {},
                         pipeline.tests_timeout_seconds if pipeline else None,
                         pipeline.tests_retries if pipeline else 0, False))
        if pipeline and pipeline.deploy_setup_cmds:
            _stages.append(("deploy_setup", pipeline.deploy_setup_cmds, pipeline.deploy_workdir,
                            pipeline.deploy_env, pipeline.deploy_timeout_seconds,
                            pipeline.deploy_retries, False))
        if pipeline and pipeline.deploy_health_cmds:
            _stages.append(("deploy_health", pipeline.deploy_health_cmds, pipeline.deploy_workdir,
                            pipeline.deploy_env, pipeline.deploy_timeout_seconds,
                            pipeline.deploy_retries, False))
        if pipeline and pipeline.rollout_run_cmds:
            _stages.append(("rollout", pipeline.rollout_run_cmds, pipeline.rollout_workdir,
                            pipeline.rollout_env, pipeline.rollout_timeout_seconds,
                            pipeline.rollout_retries, False))

        for stage_name, cmds, workdir_spec, stage_env, timeout, retries, interactive in _stages:
            res = _resolve_and_run_stage(
                repo, stage_name=stage_name, cmds=cmds, workdir_spec=workdir_spec,
                stage_env=stage_env, env_base=env_base, timeout_seconds=timeout,
                retries=retries, interactive=interactive, unattended=unattended,
                pipeline=pipeline, artifacts_dir=artifacts_dir,
            )
            stage_results[stage_name] = res
            if not res.ok:
                return _build_result(ok=False, failed_stage=stage_name,
                                     stage_results=stage_results, metrics_errors=metrics_errors)

        # --- Evaluation stage (with hints validation and metrics) ---
        if pipeline and pipeline.evaluation_run_cmds:
            eval_res = _resolve_and_run_stage(
                repo, stage_name="evaluation", cmds=pipeline.evaluation_run_cmds,
                workdir_spec=pipeline.evaluation_workdir, stage_env=pipeline.evaluation_env,
                env_base=env_base, timeout_seconds=pipeline.evaluation_timeout_seconds,
                retries=pipeline.evaluation_retries, interactive=False,
                unattended=unattended, pipeline=pipeline, artifacts_dir=artifacts_dir,
            )
            stage_results["evaluation"] = eval_res
            if not eval_res.ok:
                return _build_result(ok=False, failed_stage="evaluation",
                                     stage_results=stage_results, metrics_errors=metrics_errors)

            # Hints validation
            eval_env = safe_env(env_base, pipeline.evaluation_env, unattended=unattended)
            if _is_truthy(eval_env.get("OPENCODE_FSM_REQUIRE_HINTS")):
                hints_err = _validate_hints(repo, eval_env, artifacts_dir, metrics_errors)
                if hints_err:
                    return _build_result(ok=False, failed_stage="evaluation",
                                         stage_results=stage_results, metrics_errors=metrics_errors)

            # Evaluation metrics
            if pipeline.evaluation_metrics_path:
                mpath_str, eval_metrics, errs, m_ok = _validate_metrics(
                    repo, pipeline, pipeline.evaluation_metrics_path,
                    list(pipeline.evaluation_required_keys or []),
                    "evaluation", artifacts_dir,
                )
                if metrics_path is None and mpath_str:
                    write_text(artifacts_dir / "metrics.json",
                               read_text_if_exists(Path(pipeline.evaluation_metrics_path).expanduser()
                                                   if Path(pipeline.evaluation_metrics_path).is_absolute()
                                                   else repo / pipeline.evaluation_metrics_path))
                if not m_ok:
                    metrics_errors.extend(errs)
                    return _build_result(ok=False, failed_stage="metrics",
                                         stage_results=stage_results,
                                         metrics_path=mpath_str, metrics=eval_metrics,
                                         metrics_errors=metrics_errors)
                if metrics_path is None:
                    metrics_path = mpath_str
                    metrics = eval_metrics

        # --- Benchmark stage (with metrics) ---
        if pipeline and pipeline.benchmark_run_cmds:
            bench_res = _resolve_and_run_stage(
                repo, stage_name="benchmark", cmds=pipeline.benchmark_run_cmds,
                workdir_spec=pipeline.benchmark_workdir, stage_env=pipeline.benchmark_env,
                env_base=env_base, timeout_seconds=pipeline.benchmark_timeout_seconds,
                retries=pipeline.benchmark_retries, interactive=False,
                unattended=unattended, pipeline=pipeline, artifacts_dir=artifacts_dir,
            )
            stage_results["benchmark"] = bench_res
            if not bench_res.ok:
                return _build_result(ok=False, failed_stage="benchmark",
                                     stage_results=stage_results, metrics_errors=metrics_errors)

            if pipeline.benchmark_metrics_path:
                mpath_str, bench_metrics, errs, m_ok = _validate_metrics(
                    repo, pipeline, pipeline.benchmark_metrics_path,
                    list(pipeline.benchmark_required_keys or []),
                    "benchmark", artifacts_dir,
                )
                if metrics_path is None and mpath_str:
                    write_text(artifacts_dir / "metrics.json",
                               read_text_if_exists(Path(pipeline.benchmark_metrics_path).expanduser()
                                                   if Path(pipeline.benchmark_metrics_path).is_absolute()
                                                   else repo / pipeline.benchmark_metrics_path))
                if not m_ok:
                    metrics_errors.extend(errs)
                    return _build_result(ok=False, failed_stage="metrics",
                                         stage_results=stage_results,
                                         metrics_path=mpath_str, metrics=bench_metrics,
                                         metrics_errors=metrics_errors)
                if metrics_path is None:
                    metrics_path = mpath_str
                    metrics = bench_metrics

        ok = True
        return _build_result(ok=True, failed_stage=None, stage_results=stage_results,
                             metrics_path=metrics_path, metrics=metrics,
                             metrics_errors=metrics_errors)
    finally:
        if kubectl_dump_enabled:
            _dump_kubectl(
                artifacts_dir / "kubectl", repo,
                namespace=(pipeline.kubectl_dump_namespace if pipeline else None),
                label_selector=(pipeline.kubectl_dump_label_selector if pipeline else None),
                include_logs=bool(pipeline and pipeline.kubectl_dump_include_logs),
            )

        do_teardown = False
        if teardown_cmds and teardown_policy != "never":
            if teardown_policy == "always":
                do_teardown = True
            elif teardown_policy == "on_success":
                do_teardown = ok
            elif teardown_policy == "on_failure":
                do_teardown = not ok
        if pipeline and do_teardown:
            deploy_env = safe_env(env_base, pipeline.deploy_env, unattended=unattended)
            try:
                deploy_workdir = resolve_workdir(repo, pipeline.deploy_workdir)
            except Exception as e:
                err = CmdResult(cmd=f"resolve_workdir {pipeline.deploy_workdir}", rc=2, stdout="", stderr=str(e), timed_out=False)
                write_cmd_artifacts(artifacts_dir, "deploy_teardown_workdir_error", err)
                write_text(artifacts_dir / "deploy_teardown_warning.txt", "skip teardown due to invalid workdir\n")
                deploy_workdir = None
            if deploy_workdir is not None:
                td = _run_stage(
                    repo, stage="deploy_teardown", cmds=teardown_cmds,
                    workdir=deploy_workdir, env=deploy_env,
                    timeout_seconds=pipeline.deploy_timeout_seconds,
                    retries=0, interactive=False, unattended=unattended,
                    pipeline=pipeline, artifacts_dir=artifacts_dir,
                )
                if td.results:
                    write_cmd_artifacts(artifacts_dir, "deploy_teardown", td.results[-1])


def _validate_hints(
    repo: Path,
    eval_env: dict[str, str],
    artifacts_dir: Path,
    metrics_errors: list[str],
) -> bool:
    """Validate hints_used.json and hints_run.json. Returns True if validation failed."""
    expected = _parse_json_str_list(eval_env.get("OPENCODE_FSM_HINT_ANCHORS_JSON"))
    repo2 = Path(repo).resolve()

    # Check hints_used.json
    path = (repo2 / ".opencode_fsm" / "hints_used.json").resolve()
    if not path.exists():
        ok_hints, hint_reason = False, f"missing_hints_used_json: {path}"
    else:
        data, err = _read_json(path)
        if err:
            ok_hints, hint_reason = False, f"hints_used_json_{err}"
        else:
            assert isinstance(data, dict)
            if data.get("ok") is not True:
                ok_hints, hint_reason = False, "hints_used.ok_not_true"
            else:
                used = data.get("used_anchors")
                if not isinstance(used, list) or not used:
                    ok_hints, hint_reason = False, "hints_used.used_anchors_missing_or_empty"
                else:
                    used_clean = [str(x).strip() for x in used if isinstance(x, str) and str(x).strip()]
                    if not used_clean:
                        ok_hints, hint_reason = False, "hints_used.used_anchors_invalid"
                    else:
                        exp = [str(x).strip() for x in (expected or []) if str(x).strip()]
                        if exp and not any(u in exp for u in used_clean):
                            ok_hints, hint_reason = False, "hints_used.no_expected_anchor"
                        else:
                            commands = data.get("commands")
                            if commands is not None and (
                                not isinstance(commands, list)
                                or not any(isinstance(x, str) and x.strip() for x in commands)
                            ):
                                ok_hints, hint_reason = False, "hints_used.commands_invalid"
                            else:
                                ok_hints, hint_reason = True, "ok"
    if not ok_hints:
        try:
            write_text(artifacts_dir / "hints_requirement_error.txt", hint_reason + "\n")
        except Exception:
            pass
        metrics_errors.append(f"evaluation.hints_requirement_failed: {hint_reason}")
        return True

    # Check hints_run.json
    path = (repo2 / ".opencode_fsm" / "hints_run.json").resolve()
    if not path.exists():
        ok_run, run_reason = False, f"missing_hints_run_json: {path}"
    else:
        data, err = _read_json(path)
        if err:
            ok_run, run_reason = False, f"hints_run_json_{err}"
        else:
            assert isinstance(data, dict)
            if data.get("ok") is not True:
                ok_run, run_reason = False, "hints_run.ok_not_true"
            else:
                chosen = data.get("chosen_command")
                if not isinstance(chosen, str) or not chosen.strip():
                    ok_run, run_reason = False, "hints_run.chosen_command_missing_or_empty"
                else:
                    executed = data.get("executed_attempts")
                    try:
                        executed_i = int(executed)
                    except Exception:
                        executed_i = 0
                    if executed_i <= 0:
                        ok_run, run_reason = False, "hints_run.executed_attempts_not_positive"
                    else:
                        score = data.get("score")
                        try:
                            score_f = float(score)
                        except Exception:
                            ok_run, run_reason = False, "hints_run.score_invalid"
                        else:
                            if score_f < 0.0 or score_f > 1.0:
                                ok_run, run_reason = False, "hints_run.score_out_of_range"
                            else:
                                ok_run, run_reason = True, "ok"
    if not ok_run:
        try:
            write_text(artifacts_dir / "hints_run_requirement_error.txt", run_reason + "\n")
        except Exception:
            pass
        metrics_errors.append(f"evaluation.hints_run_requirement_failed: {run_reason}")
        return True

    return False


def fmt_stage_tail(prefix: str, stage: StageResult | None) -> str:
    res = stage_failed_cmd(stage)
    if res is None:
        return ""
    return (
        f"[{prefix}_RC]\n{res.rc}\n\n"
        f"[{prefix}_STDOUT_TAIL]\n{tail(res.stdout, STDIO_TAIL_CHARS)}\n\n"
        f"[{prefix}_STDERR_TAIL]\n{tail(res.stderr, STDIO_TAIL_CHARS)}\n\n"
    )
