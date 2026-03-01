from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..core.bootstrap import load_bootstrap_spec_with_diagnostics
from .hints import suggest_contract_hints
from ..utils.eval_audit import audit_eval_script_for_hardcoded_nonzero_score, audit_eval_script_has_real_execution
from ..core.pipeline_spec import PipelineSpec

@dataclass(frozen=True)
class ScaffoldValidationReport:
    """Structured scaffold validation output used by scaffold + repair loops."""

    missing_fields: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    stage_script_errors: list[str] = field(default_factory=list)
    bootstrap_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [
            *list(self.missing_fields or []),
            *[f"missing_file: {x}" for x in (self.missing_files or [])],
            *[f"stage_script_error: {x}" for x in (self.stage_script_errors or [])],
            *[f"bootstrap_error: {x}" for x in (self.bootstrap_errors or [])],
        ]

_BOOTSTRAP_STAGE_CMD_RE = re.compile(
    r"(?i)(^|\s)(pytest|nose2?|tox|make\s+test|runner\.generic_evaluation|"
    r"\.opencode_fsm/stages/(evaluation|rollout)\.sh)\b"
)

def validate_scaffolded_pipeline(
    pipeline: PipelineSpec,
    *,
    require_metrics: bool,
) -> list[str]:
    missing: list[str] = []

    if pipeline.security_max_cmd_seconds is None or int(pipeline.security_max_cmd_seconds) <= 0:
        missing.append("security.max_cmd_seconds")

    if not list(pipeline.deploy_setup_cmds or []):
        missing.append("deploy.setup_cmds")

    # health_cmds 允许为空（某些 repo 无健康检查），但 deploy 必须至少有 setup。

    if not list(pipeline.rollout_run_cmds or []):
        missing.append("rollout.run_cmds")

    if not list(pipeline.evaluation_run_cmds or []):
        missing.append("evaluation.run_cmds")

    if require_metrics:
        required_keys = {"score", "ok"}
        if not str(pipeline.evaluation_metrics_path or "").strip():
            missing.append("evaluation.metrics_path")
        if not required_keys.issubset(set(pipeline.evaluation_required_keys or [])):
            missing.append("evaluation.required_keys (missing: score, ok)")

    return missing

def validate_scaffolded_files(repo_root: Path) -> list[str]:
    repo_root = Path(repo_root).resolve()
    required = [
        repo_root / ".opencode_fsm" / "stages" / "tests.sh",
        repo_root / ".opencode_fsm" / "stages" / "deploy_setup.sh",
        repo_root / ".opencode_fsm" / "stages" / "deploy_health.sh",
        repo_root / ".opencode_fsm" / "stages" / "deploy_teardown.sh",
        repo_root / ".opencode_fsm" / "stages" / "rollout.sh",
        repo_root / ".opencode_fsm" / "stages" / "evaluation.sh",
        repo_root / ".opencode_fsm" / "stages" / "benchmark.sh",
    ]
    missing: list[str] = []
    for p in required:
        if not p.exists():
            missing.append(str(p.relative_to(repo_root)))
    return missing

def validate_scaffolded_stage_scripts(repo_root: Path) -> list[str]:
    """Best-effort stage script lint: currently shell syntax checks for required scripts."""
    repo_root = Path(repo_root).resolve()
    required = [
        repo_root / ".opencode_fsm" / "stages" / "tests.sh",
        repo_root / ".opencode_fsm" / "stages" / "deploy_setup.sh",
        repo_root / ".opencode_fsm" / "stages" / "deploy_health.sh",
        repo_root / ".opencode_fsm" / "stages" / "deploy_teardown.sh",
        repo_root / ".opencode_fsm" / "stages" / "rollout.sh",
        repo_root / ".opencode_fsm" / "stages" / "evaluation.sh",
        repo_root / ".opencode_fsm" / "stages" / "benchmark.sh",
    ]
    issues: list[str] = []
    for p in required:
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        if "2&&1" in text:
            issues.append(f"{p.relative_to(repo_root)}:invalid_redirect: use 2>&1 (found 2&&1)")
        try:
            res = subprocess.run(
                ["bash", "-n", str(p)],
                check=False,
                capture_output=True,
                text=True,
                timeout=4,
            )
        except Exception as e:
            issues.append(f"{p.relative_to(repo_root)}:bash_syntax_check_failed:{e}")
            continue
        if int(res.returncode) != 0:
            err = (res.stderr or res.stdout or "").strip()
            if len(err) > 500:
                err = err[-500:]
            issues.append(f"{p.relative_to(repo_root)}:{err or f'bash -n rc={res.returncode}'}")

    # Additional semantic audits for evaluation.sh: we want to reject obvious proxy
    # contracts early so OpenCode retries before expensive deploy/rollout runs.
    eval_sh = (repo_root / ".opencode_fsm" / "stages" / "evaluation.sh").resolve()
    if eval_sh.exists():
        try:
            hardcoded = audit_eval_script_for_hardcoded_nonzero_score(repo_root)
            if hardcoded:
                msg = hardcoded.strip()
                if len(msg) > 800:
                    msg = msg[:800] + "..."
                issues.append(msg)
        except Exception:
            pass
        try:
            no_exec = audit_eval_script_has_real_execution(repo_root, extra_markers=None)
            if no_exec:
                msg = str(no_exec).strip()
                if len(msg) > 500:
                    msg = msg[:500] + "..."
                issues.append(f".opencode_fsm/stages/evaluation.sh:{msg}")
        except Exception:
            pass
        try:
            hints = suggest_contract_hints(repo_root)
            if hints.commands:
                low = eval_sh.read_text(encoding="utf-8", errors="replace").lower()
                if (
                    "generic_evaluation.py" not in low
                    and "runner.generic_evaluation" not in low
                    and "hints_used.json" not in low
                ):
                    issues.append(
                        ".opencode_fsm/stages/evaluation.sh:missing_hints_provenance: "
                        "when repo hints exist, evaluation should call generic_evaluation.py or write .opencode_fsm/hints_used.json"
                    )
        except Exception:
            pass
    return issues

def validate_scaffolded_bootstrap(repo_root: Path) -> tuple[list[str], list[str]]:
    """Validate optional `.opencode_fsm/bootstrap.yml` parseability + basic contract hints."""
    repo_root = Path(repo_root).resolve()
    path = (repo_root / ".opencode_fsm" / "bootstrap.yml").resolve()
    if not path.exists():
        return [], []
    errors: list[str] = []
    warnings: list[str] = []
    try:
        loaded = load_bootstrap_spec_with_diagnostics(path)
    except Exception as e:
        return [str(e)], []
    if loaded.warnings:
        warnings.extend([f"bootstrap_warning: {w}" for w in loaded.warnings])
    for cmd in loaded.spec.cmds:
        if _BOOTSTRAP_STAGE_CMD_RE.search(str(cmd or "")):
            warnings.append(f"bootstrap_contains_stage_like_command: {cmd}")
    return errors, warnings

def validate_scaffold_contract(
    repo_root: Path,
    *,
    pipeline: PipelineSpec,
    require_metrics: bool,
) -> ScaffoldValidationReport:
    """Run full scaffold contract validation used by scaffold + repair entrypoints."""
    missing_fields = validate_scaffolded_pipeline(pipeline, require_metrics=require_metrics)
    missing_files = validate_scaffolded_files(repo_root)
    stage_script_errors = validate_scaffolded_stage_scripts(repo_root)
    bootstrap_errors, bootstrap_warnings = validate_scaffolded_bootstrap(repo_root)
    return ScaffoldValidationReport(
        missing_fields=missing_fields,
        missing_files=missing_files,
        stage_script_errors=stage_script_errors,
        bootstrap_errors=bootstrap_errors,
        warnings=bootstrap_warnings,
    )
