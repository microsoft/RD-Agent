from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from pathlib import Path

from .bootstrap import run_bootstrap
from .pipeline_spec import PipelineSpec
from .pipeline_verify import run_pipeline_verification
from .stage_cache import (
    load_stage_cache,
    save_stage_cache,
    invalidate_stage_cache,
    _file_hash,
)
from ..dtypes import VerificationResult
from .env_setup import (
    EnvHandle,
    RolloutCallResult,
    EvaluationCallResult,
    DeployCallResult,
    _default_artifacts_dir,
)

_log = logging.getLogger(__name__)


def _run_with_bootstrap(
    env: EnvHandle,
    pipeline: PipelineSpec,
    *,
    artifacts_dir: Path,
    unattended: str,
    run_bootstrap_first: bool,
    use_cache: bool = False,
) -> VerificationResult:
    """Run pipeline verification, optionally preceded by bootstrap."""
    bootstrap_path = (env.repo / ".opencode_fsm" / "bootstrap.yml").resolve()
    if run_bootstrap_first and bootstrap_path.exists():
        applied_env: dict[str, str] | None = None
        bootstrap_stage = None

        if use_cache:
            cached = load_stage_cache(env.repo, "bootstrap")
            if cached is not None:
                applied_env = cached.extra.get("applied_env")
                if isinstance(applied_env, dict):
                    _log.info("stage_cache: bootstrap HIT — skipping execution")
                else:
                    applied_env = None

        if applied_env is None:
            bootstrap_stage, applied_env = run_bootstrap(
                env.repo,
                bootstrap_path=bootstrap_path,
                pipeline=pipeline,
                unattended=str(unattended or "strict"),
                artifacts_dir=artifacts_dir / "bootstrap",
            )
            if not bootstrap_stage.ok:
                invalidate_stage_cache(env.repo, "bootstrap")
                return VerificationResult(ok=False, failed_stage="bootstrap", bootstrap=bootstrap_stage, metrics_errors=[])
            if use_cache:
                save_stage_cache(
                    env.repo, "bootstrap",
                    applied_env={str(k): str(v) for k, v in (applied_env or {}).items()},
                    bootstrap_hash=_file_hash(bootstrap_path),
                )

        old_values = {str(k): os.environ.get(str(k)) for k in (applied_env or {}).keys()}
        try:
            for k, v in (applied_env or {}).items():
                os.environ[str(k)] = str(v)
            verify = run_pipeline_verification(
                env.repo,
                pipeline=pipeline,
                tests_cmds=["echo tests_skipped"],
                artifacts_dir=artifacts_dir,
                unattended=str(unattended or "strict"),
            )
            if bootstrap_stage is not None:
                return replace(verify, bootstrap=bootstrap_stage)
            return verify
        finally:
            for k, old in old_values.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old
    else:
        return run_pipeline_verification(
            env.repo,
            pipeline=pipeline,
            tests_cmds=["echo tests_skipped"],
            artifacts_dir=artifacts_dir,
            unattended=str(unattended or "strict"),
        )


def _merge_env(base: dict[str, str], overrides: dict[str, str] | None) -> dict[str, str]:
    out = dict(base or {})
    if overrides:
        out.update({str(k): str(v) for k, v in overrides.items()})
    return out


def _stage_only_pipeline(
    pipeline: PipelineSpec,
    *,
    deploy: bool = False,
    rollout: bool,
    evaluation: bool,
    benchmark: bool,
    env_overrides: dict[str, str] | None,
) -> PipelineSpec:
    env_overrides = env_overrides or {}
    return PipelineSpec(
        security_mode=str(pipeline.security_mode or "safe"),
        security_allowlist=list(pipeline.security_allowlist or []),
        security_denylist=list(pipeline.security_denylist or []),
        security_max_cmd_seconds=pipeline.security_max_cmd_seconds,
        security_max_total_seconds=pipeline.security_max_total_seconds,
        deploy_setup_cmds=list(pipeline.deploy_setup_cmds or []) if deploy else [],
        deploy_health_cmds=list(pipeline.deploy_health_cmds or []) if deploy else [],
        deploy_teardown_cmds=list(pipeline.deploy_teardown_cmds or []) if deploy else [],
        deploy_timeout_seconds=pipeline.deploy_timeout_seconds if deploy else None,
        deploy_retries=int(pipeline.deploy_retries or 0) if deploy else 0,
        deploy_env=_merge_env(pipeline.deploy_env, env_overrides) if deploy else {},
        deploy_workdir=pipeline.deploy_workdir if deploy else None,
        deploy_teardown_policy=("on_failure" if deploy else str(pipeline.deploy_teardown_policy or "always")),
        rollout_run_cmds=list(pipeline.rollout_run_cmds or []) if rollout else [],
        rollout_timeout_seconds=pipeline.rollout_timeout_seconds if rollout else None,
        rollout_retries=int(pipeline.rollout_retries or 0) if rollout else 0,
        rollout_env=_merge_env(pipeline.rollout_env, env_overrides) if rollout else {},
        rollout_workdir=pipeline.rollout_workdir if rollout else None,
        evaluation_run_cmds=list(pipeline.evaluation_run_cmds or []) if evaluation else [],
        evaluation_timeout_seconds=pipeline.evaluation_timeout_seconds if evaluation else None,
        evaluation_retries=int(pipeline.evaluation_retries or 0) if evaluation else 0,
        evaluation_env=_merge_env(pipeline.evaluation_env, env_overrides) if evaluation else {},
        evaluation_workdir=pipeline.evaluation_workdir if evaluation else None,
        evaluation_metrics_path=pipeline.evaluation_metrics_path if evaluation else None,
        evaluation_required_keys=list(pipeline.evaluation_required_keys or []) if evaluation else [],
        benchmark_run_cmds=list(pipeline.benchmark_run_cmds or []) if benchmark else [],
        benchmark_timeout_seconds=pipeline.benchmark_timeout_seconds if benchmark else None,
        benchmark_retries=int(pipeline.benchmark_retries or 0) if benchmark else 0,
        benchmark_env=_merge_env(pipeline.benchmark_env, env_overrides) if benchmark else {},
        benchmark_workdir=pipeline.benchmark_workdir if benchmark else None,
        benchmark_metrics_path=pipeline.benchmark_metrics_path if benchmark else None,
        benchmark_required_keys=list(pipeline.benchmark_required_keys or []) if benchmark else [],
    )


def _run_deploy_health_check(env: EnvHandle, pipeline: PipelineSpec, artifacts_dir: Path, unattended: str) -> bool:
    """Execute deploy_health_cmds to verify a cached deploy is still alive."""
    if not pipeline.deploy_health_cmds:
        return True
    skip = os.environ.get("OPENCODE_FSM_CACHE_SKIP_HEALTH", "0")
    if str(skip).strip().lower() in ("1", "true", "yes"):
        return True
    health_pipeline = _stage_only_pipeline(
        pipeline, deploy=True, rollout=False, evaluation=False, benchmark=False, env_overrides=None,
    )
    health_pipeline = PipelineSpec(
        security_mode=health_pipeline.security_mode,
        security_allowlist=health_pipeline.security_allowlist,
        security_denylist=health_pipeline.security_denylist,
        security_max_cmd_seconds=health_pipeline.security_max_cmd_seconds,
        security_max_total_seconds=health_pipeline.security_max_total_seconds,
        deploy_setup_cmds=[],
        deploy_health_cmds=list(pipeline.deploy_health_cmds or []),
        deploy_teardown_cmds=[],
        deploy_timeout_seconds=pipeline.deploy_timeout_seconds,
        deploy_retries=0,
        deploy_env=health_pipeline.deploy_env,
        deploy_workdir=pipeline.deploy_workdir,
        deploy_teardown_policy="never",
    )
    health_dir = (artifacts_dir / "cache_health_check").resolve()
    health_dir.mkdir(parents=True, exist_ok=True)
    result = run_pipeline_verification(
        env.repo, pipeline=health_pipeline,
        tests_cmds=["echo tests_skipped"],
        artifacts_dir=health_dir, unattended=unattended,
    )
    deploy_health = getattr(result, "deploy_health", None)
    return bool(deploy_health and deploy_health.ok) if deploy_health is not None else bool(result.ok)


def rollout(
    env: EnvHandle,
    *,
    artifacts_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    unattended: str = "strict",
    run_bootstrap_first: bool = True,
    use_cache: bool = False,
) -> RolloutCallResult:
    artifacts_dir = (artifacts_dir or _default_artifacts_dir(env.repo, prefix="rollout")).resolve()
    p = _stage_only_pipeline(env.pipeline, deploy=False, rollout=True, evaluation=False, benchmark=False, env_overrides=env_overrides)
    verify = _run_with_bootstrap(env, p, artifacts_dir=artifacts_dir, unattended=unattended, run_bootstrap_first=run_bootstrap_first, use_cache=use_cache)
    if verify.failed_stage == "bootstrap":
        return RolloutCallResult(ok=False, artifacts_dir=artifacts_dir, rollout_path=None, verify=verify)
    rollout_path = (env.repo / ".opencode_fsm" / "rollout.json").resolve()
    if not rollout_path.exists():
        rollout_path = None
    ok = bool(verify.ok)
    if ok and use_cache and rollout_path:
        save_stage_cache(env.repo, "rollout", rollout_path=str(rollout_path))
    elif not ok:
        invalidate_stage_cache(env.repo, "rollout")
    return RolloutCallResult(ok=ok, artifacts_dir=artifacts_dir, rollout_path=rollout_path, verify=verify)


def evaluate(
    env: EnvHandle,
    *,
    artifacts_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    unattended: str = "strict",
    run_bootstrap_first: bool = True,
    use_cache: bool = False,
) -> EvaluationCallResult:
    artifacts_dir = (artifacts_dir or _default_artifacts_dir(env.repo, prefix="evaluation")).resolve()
    p = _stage_only_pipeline(env.pipeline, deploy=False, rollout=False, evaluation=True, benchmark=False, env_overrides=env_overrides)
    verify = _run_with_bootstrap(env, p, artifacts_dir=artifacts_dir, unattended=unattended, run_bootstrap_first=run_bootstrap_first, use_cache=use_cache)
    if verify.failed_stage == "bootstrap":
        return EvaluationCallResult(ok=False, artifacts_dir=artifacts_dir, metrics_path=None, metrics=None, verify=verify)
    metrics_path = Path(str(verify.metrics_path)).resolve() if getattr(verify, "metrics_path", None) else None
    metrics = getattr(verify, "metrics", None)
    return EvaluationCallResult(
        ok=bool(verify.ok), artifacts_dir=artifacts_dir,
        metrics_path=metrics_path, metrics=metrics, verify=verify,
    )


def rollout_and_evaluate(
    env: EnvHandle,
    *,
    artifacts_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    unattended: str = "strict",
    run_bootstrap_first: bool = True,
    use_cache: bool = False,
) -> tuple[RolloutCallResult, EvaluationCallResult]:
    artifacts_dir = (artifacts_dir or _default_artifacts_dir(env.repo, prefix="rollout_evaluation")).resolve()
    p = _stage_only_pipeline(env.pipeline, deploy=False, rollout=True, evaluation=True, benchmark=False, env_overrides=env_overrides)
    verify = _run_with_bootstrap(env, p, artifacts_dir=artifacts_dir, unattended=unattended, run_bootstrap_first=run_bootstrap_first, use_cache=use_cache)
    if verify.failed_stage == "bootstrap":
        return (
            RolloutCallResult(ok=False, artifacts_dir=artifacts_dir, rollout_path=None, verify=verify),
            EvaluationCallResult(ok=False, artifacts_dir=artifacts_dir, metrics_path=None, metrics=None, verify=verify),
        )

    rollout_path = (env.repo / ".opencode_fsm" / "rollout.json").resolve()
    if not rollout_path.exists():
        rollout_path = None
    metrics_path = Path(str(verify.metrics_path)).resolve() if getattr(verify, "metrics_path", None) else None
    metrics = getattr(verify, "metrics", None)

    rollout_stage = getattr(verify, "rollout", None)
    rollout_stage_ok = bool(getattr(rollout_stage, "ok", False)) if rollout_stage is not None else bool(verify.ok)
    return (
        RolloutCallResult(ok=rollout_stage_ok, artifacts_dir=artifacts_dir, rollout_path=rollout_path, verify=verify),
        EvaluationCallResult(
            ok=bool(verify.ok), artifacts_dir=artifacts_dir,
            metrics_path=metrics_path, metrics=metrics, verify=verify,
        ),
    )


def with_runtime_env_path(runtime_env_path: str | Path) -> dict[str, str]:
    p = Path(str(runtime_env_path)).expanduser().resolve()
    return _merge_env({}, {"OPENCODE_RUNTIME_ENV_PATH": str(p)})


def deploy(
    env: EnvHandle,
    *,
    artifacts_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    unattended: str = "strict",
    run_bootstrap_first: bool = True,
    runtime_env_rel: str = ".opencode_fsm/runtime_env.json",
    use_cache: bool = False,
) -> DeployCallResult:
    artifacts_dir = (artifacts_dir or _default_artifacts_dir(env.repo, prefix="deploy")).resolve()

    if use_cache:
        cached = load_stage_cache(env.repo, "deploy")
        if cached is not None:
            cached_rp = cached.extra.get("runtime_env_path")
            if isinstance(cached_rp, str) and cached_rp and Path(cached_rp).exists():
                health_ok = _run_deploy_health_check(env, env.pipeline, artifacts_dir, unattended)
                if health_ok:
                    _log.info("stage_cache: deploy HIT + health OK — skipping execution")
                    runtime_env = None
                    try:
                        runtime_env = json.loads(Path(cached_rp).read_text(encoding="utf-8", errors="replace"))
                        if not isinstance(runtime_env, dict):
                            runtime_env = None
                    except Exception:
                        runtime_env = None
                    return DeployCallResult(
                        ok=True, artifacts_dir=artifacts_dir,
                        runtime_env_path=Path(cached_rp), runtime_env=runtime_env, verify=None,
                    )
                else:
                    _log.info("stage_cache: deploy health check FAILED — re-deploying")
                    invalidate_stage_cache(env.repo, "deploy")

    p = _stage_only_pipeline(env.pipeline, deploy=True, rollout=False, evaluation=False, benchmark=False, env_overrides=env_overrides)
    verify = _run_with_bootstrap(env, p, artifacts_dir=artifacts_dir, unattended=unattended, run_bootstrap_first=run_bootstrap_first, use_cache=use_cache)
    if verify.failed_stage == "bootstrap":
        return DeployCallResult(ok=False, artifacts_dir=artifacts_dir, runtime_env_path=None, runtime_env=None, verify=verify)

    runtime_env_path = (env.repo / str(runtime_env_rel)).resolve()
    runtime_env = None
    if runtime_env_path.exists():
        try:
            runtime_env = json.loads(runtime_env_path.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(runtime_env, dict):
                runtime_env = None
        except Exception:
            runtime_env = None
    else:
        runtime_env_path = None

    ok = bool(verify.ok)
    if ok and use_cache and runtime_env_path:
        save_stage_cache(env.repo, "deploy", runtime_env_path=str(runtime_env_path))
    elif not ok:
        invalidate_stage_cache(env.repo, "deploy")

    return DeployCallResult(
        ok=ok, artifacts_dir=artifacts_dir,
        runtime_env_path=runtime_env_path, runtime_env=runtime_env, verify=verify,
    )


def deploy_teardown(
    env: EnvHandle,
    *,
    artifacts_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    unattended: str = "strict",
) -> bool:
    """Best-effort run of deploy teardown commands."""
    if not list(env.pipeline.deploy_teardown_cmds or []):
        return True

    artifacts_dir = (artifacts_dir or _default_artifacts_dir(env.repo, prefix="deploy_teardown")).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    p = PipelineSpec(
        security_mode=str(env.pipeline.security_mode or "safe"),
        security_allowlist=list(env.pipeline.security_allowlist or []),
        security_denylist=list(env.pipeline.security_denylist or []),
        security_max_cmd_seconds=env.pipeline.security_max_cmd_seconds,
        security_max_total_seconds=env.pipeline.security_max_total_seconds,
        deploy_setup_cmds=[],
        deploy_health_cmds=[],
        deploy_teardown_cmds=list(env.pipeline.deploy_teardown_cmds or []),
        deploy_timeout_seconds=env.pipeline.deploy_timeout_seconds,
        deploy_retries=0,
        deploy_env=_merge_env(env.pipeline.deploy_env, env_overrides),
        deploy_workdir=env.pipeline.deploy_workdir,
        deploy_teardown_policy="always",
    )

    run_pipeline_verification(
        env.repo,
        pipeline=p,
        tests_cmds=["echo tests_skipped"],
        artifacts_dir=artifacts_dir,
        unattended=str(unattended or "strict"),
    )

    summary_path = (artifacts_dir / "deploy_teardown_summary.json").resolve()
    if not summary_path.exists():
        return False
    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8", errors="replace"))
        return bool(isinstance(obj, dict) and obj.get("ok") is True)
    except Exception:
        return False
