from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.subprocess import read_text_if_exists

@dataclass(frozen=True)
class PipelineSpec:

    version: int = 1

    # tests
    tests_cmds: list[str] = None  # type: ignore[assignment]
    tests_timeout_seconds: int | None = None
    tests_retries: int = 0
    tests_env: dict[str, str] = None  # type: ignore[assignment]
    tests_workdir: str | None = None

    # deploy
    deploy_setup_cmds: list[str] = None  # type: ignore[assignment]
    deploy_health_cmds: list[str] = None  # type: ignore[assignment]
    deploy_teardown_cmds: list[str] = None  # type: ignore[assignment]
    deploy_timeout_seconds: int | None = None
    deploy_retries: int = 0
    deploy_env: dict[str, str] = None  # type: ignore[assignment]
    deploy_workdir: str | None = None
    deploy_teardown_policy: str = "always"  # always|on_success|on_failure|never

    kubectl_dump_enabled: bool = False
    kubectl_dump_namespace: str | None = None
    kubectl_dump_label_selector: str | None = None
    kubectl_dump_include_logs: bool = False

    # rollout (optional)
    rollout_run_cmds: list[str] = None  # type: ignore[assignment]
    rollout_timeout_seconds: int | None = None
    rollout_retries: int = 0
    rollout_env: dict[str, str] = None  # type: ignore[assignment]
    rollout_workdir: str | None = None

    # evaluation (optional; preferred over benchmark for "evaluation metrics")
    evaluation_run_cmds: list[str] = None  # type: ignore[assignment]
    evaluation_timeout_seconds: int | None = None
    evaluation_retries: int = 0
    evaluation_env: dict[str, str] = None  # type: ignore[assignment]
    evaluation_workdir: str | None = None
    evaluation_metrics_path: str | None = None
    evaluation_required_keys: list[str] = None  # type: ignore[assignment]

    # benchmark
    benchmark_run_cmds: list[str] = None  # type: ignore[assignment]
    benchmark_timeout_seconds: int | None = None
    benchmark_retries: int = 0
    benchmark_env: dict[str, str] = None  # type: ignore[assignment]
    benchmark_workdir: str | None = None
    benchmark_metrics_path: str | None = None
    benchmark_required_keys: list[str] = None  # type: ignore[assignment]

    # auth (optional)
    auth_cmds: list[str] = None  # type: ignore[assignment]
    auth_timeout_seconds: int | None = None
    auth_retries: int = 0
    auth_env: dict[str, str] = None  # type: ignore[assignment]
    auth_workdir: str | None = None
    auth_interactive: bool = False

    # artifacts
    artifacts_out_dir: str | None = None

    # tooling (deprecated; prefer .opencode_fsm/actions.yml)
    tooling_ensure_tools: bool = False
    tooling_ensure_kind_cluster: bool = False
    tooling_kind_cluster_name: str = "kind"
    tooling_kind_config: str | None = None

    # security
    security_mode: str = "safe"  # safe|system
    security_allowlist: list[str] = None  # type: ignore[assignment]
    security_denylist: list[str] = None  # type: ignore[assignment]
    security_max_cmd_seconds: int | None = None
    security_max_total_seconds: int | None = None

    def __post_init__(self) -> None:
        for attr in (
            "tests_cmds",
            "deploy_setup_cmds",
            "deploy_health_cmds",
            "deploy_teardown_cmds",
            "rollout_run_cmds",
            "evaluation_run_cmds",
            "benchmark_run_cmds",
            "evaluation_required_keys",
            "benchmark_required_keys",
            "auth_cmds",
            "security_allowlist",
            "security_denylist",
        ):
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, [])
        for attr in ("tests_env", "deploy_env", "rollout_env", "evaluation_env", "benchmark_env", "auth_env"):
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, {})

def load_pipeline_spec(path: Path) -> PipelineSpec:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: PyYAML. Install with `pip install PyYAML`.") from e

    raw = read_text_if_exists(path).strip()
    if not raw:
        raise ValueError(f"pipeline file is empty: {path}")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("pipeline must be a YAML mapping (dict) at the top level")

    version = int(data.get("version") or 1)
    if version != 1:
        raise ValueError(f"unsupported pipeline version: {version}")

    mappings: dict[str, dict[str, Any]] = {}
    for key, name in (
        ("tests", "tests"),
        ("deploy", "deploy"),
        ("rollout", "rollout"),
        ("evaluation", "evaluation"),
        ("benchmark", "benchmark"),
        ("artifacts", "artifacts"),
        ("tooling", "tooling"),
        ("auth", "auth"),
        ("security", "security"),
    ):
        v = data.get(key)
        if v is None:
            mappings[key] = {}
        elif not isinstance(v, dict):
            raise ValueError(f"pipeline.{name} must be a mapping")
        else:
            mappings[key] = v

    tests = mappings["tests"]
    deploy = mappings["deploy"]
    rollout = mappings["rollout"]
    evaluation = mappings["evaluation"]
    bench = mappings["benchmark"]
    artifacts = mappings["artifacts"]
    tooling = mappings["tooling"]
    auth = mappings["auth"]
    security = mappings["security"]

    v = deploy.get("kubectl_dump")
    if v is None:
        kubectl_dump: dict[str, Any] = {}
    elif not isinstance(v, dict):
        raise ValueError("pipeline.deploy.kubectl_dump must be a mapping")
    else:
        kubectl_dump = v

    eval_required_keys = evaluation.get("required_keys") or []
    if eval_required_keys is None:
        eval_required_keys = []
    if not isinstance(eval_required_keys, list) or not all(isinstance(k, str) for k in eval_required_keys):
        raise ValueError("pipeline.evaluation.required_keys must be a list of strings")

    required_keys = bench.get("required_keys") or []
    if required_keys is None:
        required_keys = []
    if not isinstance(required_keys, list) or not all(isinstance(k, str) for k in required_keys):
        raise ValueError("pipeline.benchmark.required_keys must be a list of strings")

    teardown_policy = str(deploy.get("teardown_policy") or "always").strip().lower()
    if teardown_policy not in ("always", "on_success", "on_failure", "never"):
        raise ValueError("pipeline.deploy.teardown_policy must be one of: always, on_success, on_failure, never")

    kind_cluster_name = str(tooling.get("kind_cluster_name") or "kind").strip() or "kind"

    security_mode = str(security.get("mode") or "safe").strip().lower()
    if security_mode not in ("safe", "system"):
        raise ValueError("pipeline.security.mode must be one of: safe, system")

    allowlist = security.get("allowlist") or []
    if allowlist is None:
        allowlist = []
    if not isinstance(allowlist, list) or not all(isinstance(x, str) for x in allowlist):
        raise ValueError("pipeline.security.allowlist must be a list of strings")

    denylist = security.get("denylist") or []
    if denylist is None:
        denylist = []
    if not isinstance(denylist, list) or not all(isinstance(x, str) for x in denylist):
        raise ValueError("pipeline.security.denylist must be a list of strings")

    parsed_cmds: dict[str, list[str]] = {}
    for out_name, m, cmd_key, cmds_key in (
        ("auth_cmds", auth, "cmd", "cmds"),
        ("tests_cmds", tests, "cmd", "cmds"),
        ("deploy_setup_cmds", deploy, "setup_cmd", "setup_cmds"),
        ("deploy_health_cmds", deploy, "health_cmd", "health_cmds"),
        ("deploy_teardown_cmds", deploy, "teardown_cmd", "teardown_cmds"),
        ("rollout_run_cmds", rollout, "run_cmd", "run_cmds"),
        ("evaluation_run_cmds", evaluation, "run_cmd", "run_cmds"),
        ("benchmark_run_cmds", bench, "run_cmd", "run_cmds"),
    ):
        if cmds_key in m and m.get(cmds_key) is not None:
            vv = m.get(cmds_key)
            if not isinstance(vv, list) or not all(isinstance(x, str) and x.strip() for x in vv):
                raise ValueError(f"pipeline field {cmds_key} must be a list of non-empty strings")
            parsed_cmds[out_name] = [x.strip() for x in vv if x.strip()]
        else:
            vv = m.get(cmd_key)
            if vv is None:
                parsed_cmds[out_name] = []
            elif not isinstance(vv, str) or not vv.strip():
                raise ValueError(f"pipeline field {cmd_key} must be a non-empty string")
            else:
                parsed_cmds[out_name] = [vv.strip()]

    parsed_envs: dict[str, dict[str, str]] = {}
    for stage_name, m in (
        ("tests", tests),
        ("deploy", deploy),
        ("rollout", rollout),
        ("evaluation", evaluation),
        ("benchmark", bench),
        ("auth", auth),
    ):
        vv = m.get("env")
        if vv is None:
            parsed_envs[stage_name] = {}
        elif not isinstance(vv, dict):
            raise ValueError(f"pipeline.{stage_name}.env must be a mapping")
        else:
            out: dict[str, str] = {}
            for k, v in vv.items():
                if k is None:
                    continue
                ks = str(k).strip()
                if not ks:
                    continue
                out[ks] = "" if v is None else str(v)
            parsed_envs[stage_name] = out

    # auth: accept cmds or steps as alias
    auth_cmds = parsed_cmds["auth_cmds"]
    if not auth_cmds and auth.get("steps") is not None:
        steps = auth.get("steps")
        if not isinstance(steps, list) or not all(isinstance(x, str) and x.strip() for x in steps):
            raise ValueError("pipeline.auth.steps must be a list of non-empty strings")
        auth_cmds = [x.strip() for x in steps if x.strip()]

    return PipelineSpec(
        version=version,
        tests_cmds=parsed_cmds["tests_cmds"],
        tests_timeout_seconds=(int(tests.get("timeout_seconds")) if tests.get("timeout_seconds") else None),
        tests_retries=int(tests.get("retries") or 0),
        tests_env=parsed_envs["tests"],
        tests_workdir=(str(tests.get("workdir")).strip() if tests.get("workdir") else None),
        deploy_setup_cmds=parsed_cmds["deploy_setup_cmds"],
        deploy_health_cmds=parsed_cmds["deploy_health_cmds"],
        deploy_teardown_cmds=parsed_cmds["deploy_teardown_cmds"],
        deploy_timeout_seconds=(int(deploy.get("timeout_seconds")) if deploy.get("timeout_seconds") else None),
        deploy_retries=int(deploy.get("retries") or 0),
        deploy_env=parsed_envs["deploy"],
        deploy_workdir=(str(deploy.get("workdir")).strip() if deploy.get("workdir") else None),
        deploy_teardown_policy=teardown_policy,
        kubectl_dump_enabled=bool(kubectl_dump.get("enabled") or False),
        kubectl_dump_namespace=(str(kubectl_dump.get("namespace")).strip() if kubectl_dump.get("namespace") else None),
        kubectl_dump_label_selector=(
            str(kubectl_dump.get("label_selector")).strip() if kubectl_dump.get("label_selector") else None
        ),
        kubectl_dump_include_logs=bool(kubectl_dump.get("include_logs") or False),
        rollout_run_cmds=parsed_cmds["rollout_run_cmds"],
        rollout_timeout_seconds=(int(rollout.get("timeout_seconds")) if rollout.get("timeout_seconds") else None),
        rollout_retries=int(rollout.get("retries") or 0),
        rollout_env=parsed_envs["rollout"],
        rollout_workdir=(str(rollout.get("workdir")).strip() if rollout.get("workdir") else None),
        evaluation_run_cmds=parsed_cmds["evaluation_run_cmds"],
        evaluation_timeout_seconds=(
            int(evaluation.get("timeout_seconds")) if evaluation.get("timeout_seconds") else None
        ),
        evaluation_retries=int(evaluation.get("retries") or 0),
        evaluation_env=parsed_envs["evaluation"],
        evaluation_workdir=(str(evaluation.get("workdir")).strip() if evaluation.get("workdir") else None),
        evaluation_metrics_path=(
            str(evaluation.get("metrics_path")).strip() if evaluation.get("metrics_path") else None
        ),
        evaluation_required_keys=[str(k).strip() for k in eval_required_keys if str(k).strip()],
        benchmark_run_cmds=parsed_cmds["benchmark_run_cmds"],
        benchmark_timeout_seconds=(int(bench.get("timeout_seconds")) if bench.get("timeout_seconds") else None),
        benchmark_retries=int(bench.get("retries") or 0),
        benchmark_env=parsed_envs["benchmark"],
        benchmark_workdir=(str(bench.get("workdir")).strip() if bench.get("workdir") else None),
        benchmark_metrics_path=(str(bench.get("metrics_path")).strip() if bench.get("metrics_path") else None),
        benchmark_required_keys=[str(k).strip() for k in required_keys if str(k).strip()],
        auth_cmds=auth_cmds,
        auth_timeout_seconds=(int(auth.get("timeout_seconds")) if auth.get("timeout_seconds") else None),
        auth_retries=int(auth.get("retries") or 0),
        auth_env=parsed_envs["auth"],
        auth_workdir=(str(auth.get("workdir")).strip() if auth.get("workdir") else None),
        auth_interactive=bool(auth.get("interactive") or False),
        artifacts_out_dir=(str(artifacts.get("out_dir")).strip() if artifacts.get("out_dir") else None),
        tooling_ensure_tools=bool(tooling.get("ensure_tools") or False),
        tooling_ensure_kind_cluster=bool(tooling.get("ensure_kind_cluster") or False),
        tooling_kind_cluster_name=kind_cluster_name,
        tooling_kind_config=(str(tooling.get("kind_config")).strip() if tooling.get("kind_config") else None),
        security_mode=security_mode,
        security_allowlist=[str(x).strip() for x in allowlist if str(x).strip()],
        security_denylist=[str(x).strip() for x in denylist if str(x).strip()],
        security_max_cmd_seconds=(int(security.get("max_cmd_seconds")) if security.get("max_cmd_seconds") else None),
        security_max_total_seconds=(int(security.get("max_total_seconds")) if security.get("max_total_seconds") else None),
    )
