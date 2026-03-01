from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..dtypes import AgentClient, VerificationResult
from ..contract.hints import suggest_contract_hints
from ..contract.provenance import (
    build_contract_provenance_report,
    dump_provenance,
    snapshot_contract_files,
)
from ..opencode.client import OpenCodeClient
from ..opencode.prompts import make_scaffold_contract_prompt, make_scaffold_contract_retry_prompt
from .pipeline_spec import PipelineSpec, load_pipeline_spec
from ..contract.validation import validate_scaffold_contract
from ..utils.subprocess import tail, write_text


@dataclass(frozen=True)
class EnvHandle:
    repo: Path
    pipeline_path: Path
    pipeline: PipelineSpec


@dataclass(frozen=True)
class RolloutCallResult:
    ok: bool
    artifacts_dir: Path
    rollout_path: Path | None
    verify: VerificationResult


@dataclass(frozen=True)
class EvaluationCallResult:
    ok: bool
    artifacts_dir: Path
    metrics_path: Path | None
    metrics: dict | None
    verify: VerificationResult


@dataclass(frozen=True)
class DeployCallResult:
    ok: bool
    artifacts_dir: Path
    runtime_env_path: Path | None
    runtime_env: dict | None
    verify: VerificationResult


def _default_artifacts_dir(repo: Path, *, prefix: str) -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return (repo / ".opencode_fsm" / "artifacts" / run_id / prefix).resolve()


def _resolve_model(raw_model: str) -> str:
    candidates: list[str] = []
    if shutil.which("opencode"):
        try:
            res = subprocess.run(
                ["opencode", "models"],
                check=False, capture_output=True, text=True, timeout=10,
            )
        except Exception:
            res = None
        if res is not None and int(res.returncode) == 0:
            ansi = re.compile(r"\x1b\[[0-9;]*m")
            for raw in (res.stdout or "").splitlines():
                line = ansi.sub("", raw).strip()
                if not line or "/" not in line:
                    continue
                candidates.append(line)

    s = str(raw_model or "").strip()
    if not s:
        env_default = str(
            os.environ.get("OPENCODE_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("CHAT_MODEL")
            or os.environ.get("LITELLM_CHAT_MODEL")
            or ""
        ).strip()
        if env_default:
            s = env_default
        else:
            if "myproxy/deepseek-v3.2" in candidates:
                return "myproxy/deepseek-v3.2"
            if "openai/deepseek-v3.2" in candidates:
                return "openai/deepseek-v3.2"
            if "openai/gpt-4o-mini" in candidates:
                return "openai/gpt-4o-mini"
            if "opencode/gpt-5-nano" in candidates:
                return "opencode/gpt-5-nano"
            if candidates:
                return candidates[0]
            return "openai/gpt-4o-mini"

    if "/" in s:
        if not candidates:
            return s
        if s in candidates:
            return s
        try:
            _provider, model_id = s.split("/", 1)
        except Exception:
            return s
        matches = [m for m in candidates if m.split("/", 1)[1] == model_id]
        if matches:
            for m in matches:
                if m.startswith("myproxy/"):
                    return m
            return matches[0]
        return s

    matches = [m for m in candidates if m.split("/", 1)[1] == s]
    if matches:
        for m in matches:
            if m.startswith("myproxy/"):
                return m
        return matches[0]
    if candidates:
        return candidates[0]
    return f"openai/{s}"


def open_env(
    repo: str | Path,
    *,
    clones_dir: Path | None = None,
    pipeline_rel: str = "pipeline.yml",
    require_pipeline: bool = True,
    scaffold_contract: str = "opencode",
    scaffold_require_metrics: bool = True,
    model: str = "",
    opencode_url: str = "",
    opencode_timeout_seconds: int = 300,
    opencode_retry_attempts: int = 2,
    opencode_retry_backoff_seconds: float = 2.0,
    opencode_session_recover_attempts: int | None = None,
    opencode_session_recover_backoff_seconds: float | None = None,
    opencode_context_length: int | None = None,
    opencode_max_prompt_chars: int | None = None,
    opencode_bash: str = "restricted",
    scaffold_opencode_bash: str = "full",
    unattended: str = "strict",
    artifacts_dir: Path | None = None,
    seed_stage_skeleton: bool = True,
    write_fallback_pipeline_yml: bool = True,
    agent: AgentClient | None = None,
    opencode_auto_compact: bool | None = None,
) -> EnvHandle:
    from ..utils.repo_resolver import prepare_repo

    prepared = prepare_repo(str(repo), clones_dir=clones_dir)
    repo_root = prepared.repo.resolve()
    pipeline_path = (repo_root / str(pipeline_rel)).resolve()
    if not pipeline_path.exists():
        mode = str(scaffold_contract or "off").strip().lower() or "off"
        if mode not in ("off", "opencode"):
            mode = "off"
        if mode == "opencode":
            out_dir = (artifacts_dir or _default_artifacts_dir(repo_root, prefix="scaffold")).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            created_agent = False
            scaffold_err = ""
            tool_trace: list[dict[str, object]] = []
            provenance_written = False
            contract_before = snapshot_contract_files(repo_root)
            runner_written_paths: set[str] = set()
            try:
                max_scaffold_attempts = int(os.environ.get("OPENCODE_SCAFFOLD_ATTEMPTS") or 5)
            except Exception:
                max_scaffold_attempts = 5
            max_scaffold_attempts = max(1, min(10, int(max_scaffold_attempts)))
            last_failure_reason = ""

            try:
                if agent is None:
                    created_agent = True
                    agent = OpenCodeClient(
                        repo=repo_root,
                        plan_rel="PLAN.md",
                        pipeline_rel=str(pipeline_rel).strip() or "pipeline.yml",
                        model=_resolve_model(model),
                        base_url=(str(opencode_url or "").strip() or None),
                        timeout_seconds=int(opencode_timeout_seconds or 300),
                        request_retry_attempts=int(opencode_retry_attempts or 0),
                        request_retry_backoff_seconds=float(opencode_retry_backoff_seconds or 0.0),
                        session_recover_attempts=(
                            int(opencode_session_recover_attempts)
                            if opencode_session_recover_attempts is not None
                            else None
                        ),
                        session_recover_backoff_seconds=(
                            float(opencode_session_recover_backoff_seconds)
                            if opencode_session_recover_backoff_seconds is not None
                            else None
                        ),
                        context_length=(int(opencode_context_length) if opencode_context_length is not None else None),
                        max_prompt_chars=(int(opencode_max_prompt_chars) if opencode_max_prompt_chars is not None else None),
                        bash_mode=str(opencode_bash or "restricted"),
                        scaffold_bash_mode=str(scaffold_opencode_bash or "full"),
                        unattended=str(unattended or "strict"),
                        auto_compact=opencode_auto_compact,
                        server_log_path=out_dir / "opencode_server.log",
                        session_title=f"{repo_root.name}:scaffold",
                        username=(str(os.environ.get("OPENCODE_SERVER_USERNAME") or "opencode").strip() or "opencode")
                        if str(opencode_url or "").strip()
                        else None,
                        password=(str(os.environ.get("OPENCODE_SERVER_PASSWORD") or "").strip() or None)
                        if str(opencode_url or "").strip()
                        else None,
                        permission_overrides={"external_directory": {"*": "allow"}},
                    )
                hints = suggest_contract_hints(repo_root)
                if hints.commands:
                    write_text(out_dir / "scaffold_command_hints.txt", "\n".join(hints.commands) + "\n")

                pipeline_ok = False
                pipeline_path = (repo_root / str(pipeline_rel).strip() or "pipeline.yml").resolve()
                for attempt in range(1, max_scaffold_attempts + 1):
                    try:
                        if attempt <= 1:
                            prompt = make_scaffold_contract_prompt(
                                repo_root,
                                pipeline_rel=str(pipeline_rel).strip() or "pipeline.yml",
                                require_metrics=bool(scaffold_require_metrics),
                                command_hints=hints.commands,
                            )
                        else:
                            prompt = make_scaffold_contract_retry_prompt(
                                repo_root,
                                pipeline_rel=str(pipeline_rel).strip() or "pipeline.yml",
                                require_metrics=bool(scaffold_require_metrics),
                                attempt=attempt,
                                max_attempts=max_scaffold_attempts,
                                previous_failure=last_failure_reason or scaffold_err,
                                command_hints=hints.commands,
                            )
                        res = agent.run(
                            prompt,
                            fsm_state="S0_SCAFFOLD",
                            iter_idx=max(0, attempt - 1),
                            purpose="scaffold_contract",
                        )
                        turn_trace = [dict(x) for x in (res.tool_trace or []) if isinstance(x, dict)]
                        for x in turn_trace:
                            x["attempt"] = int(attempt)
                        tool_trace.extend(turn_trace)
                        write_text(
                            out_dir / f"scaffold_agent_result_attempt_{attempt:02d}.txt",
                            tail(res.assistant_text or "", 20000) + "\n",
                        )
                        write_text(out_dir / "scaffold_agent_result.txt", tail(res.assistant_text or "", 20000) + "\n")
                    except Exception as e:
                        scaffold_err = tail(str(e), 4000)
                        write_text(out_dir / f"scaffold_agent_error_attempt_{attempt:02d}.txt", scaffold_err + "\n")
                        write_text(out_dir / "scaffold_agent_error.txt", scaffold_err + "\n")

                    if not pipeline_path.exists():
                        low = str(scaffold_err or "").strip().lower()
                        if low and any(
                            n in low
                            for n in (
                                "connection refused", "failed to establish a new connection",
                                "connection reset", "connection aborted",
                                "timed out", "timeout", "network is unreachable",
                                "temporary failure in name resolution", "name or service not known",
                            )
                        ):
                            transport_reason = "opencode_transport_unavailable"
                        else:
                            transport_reason = ""
                        if transport_reason:
                            last_failure_reason = f"missing_pipeline_yml; {transport_reason}"
                        else:
                            last_failure_reason = "missing_pipeline_yml"
                        continue

                    try:
                        parsed = load_pipeline_spec(pipeline_path)
                    except Exception as e:
                        last_failure_reason = f"pipeline_parse_error: {tail(str(e), 1000)}"
                        write_text(
                            out_dir / f"scaffold_agent_pipeline_parse_error_attempt_{attempt:02d}.txt",
                            tail(str(e), 4000) + "\n",
                        )
                        write_text(out_dir / "scaffold_agent_pipeline_parse_error.txt", tail(str(e), 4000) + "\n")
                        continue

                    report = validate_scaffold_contract(
                        repo_root, pipeline=parsed, require_metrics=bool(scaffold_require_metrics),
                    )
                    if not report.errors:
                        pipeline_ok = True
                        if report.warnings:
                            write_text(
                                out_dir / "scaffold_validation_warning.txt",
                                "\n".join([f"- {x}" for x in report.warnings]) + "\n",
                            )
                        break

                    last_failure_reason = "incomplete_contract: " + "; ".join(report.errors or [])
                    write_text(
                        out_dir / f"scaffold_agent_pipeline_validation_error_attempt_{attempt:02d}.txt",
                        "Pipeline is parseable but does not meet scaffold requirements:\n"
                        + "\n".join([f"- {x}" for x in report.errors])
                        + ("\n" if report.errors else "")
                        + (
                            "Non-fatal warnings:\n"
                            + "\n".join([f"- {x}" for x in (report.warnings or [])])
                            + ("\n" if report.warnings else "")
                            if report.warnings
                            else ""
                        ),
                    )
                    write_text(
                        out_dir / "scaffold_agent_pipeline_validation_error.txt",
                        "Pipeline is parseable but does not meet scaffold requirements:\n"
                        + "\n".join([f"- {x}" for x in report.errors])
                        + ("\n" if report.errors else ""),
                    )

                if not pipeline_ok:
                    low = str(scaffold_err or "").strip().lower()
                    if low and any(
                        n in low
                        for n in (
                            "connection refused", "failed to establish a new connection",
                            "connection reset", "connection aborted",
                            "timed out", "timeout", "network is unreachable",
                            "temporary failure in name resolution", "name or service not known",
                        )
                    ):
                        root_cause = "opencode_transport_unavailable"
                    else:
                        root_cause = ""
                    if not provenance_written:
                        try:
                            report = build_contract_provenance_report(
                                repo=repo_root, purpose="scaffold_contract",
                                strict_opencode=(not bool(seed_stage_skeleton) and not bool(write_fallback_pipeline_yml)),
                                before=contract_before, after=snapshot_contract_files(repo_root),
                                tool_trace=tool_trace, runner_written_paths=runner_written_paths,
                            )
                            dump_provenance(out_dir / "scaffold_provenance.json", report)
                            provenance_written = True
                        except Exception:
                            pass
                    write_text(
                        out_dir / "scaffold_error.txt",
                        "scaffold_contract_failed: missing_or_invalid_pipeline_yml\n"
                        + f"attempts: {max_scaffold_attempts}\n"
                        + (f"last_failure: {last_failure_reason}\n" if last_failure_reason else "")
                        + (f"root_cause: {root_cause}\n" if root_cause else "")
                        + (f"agent_error: {scaffold_err}\n" if scaffold_err else ""),
                    )
                    raise RuntimeError(
                        f"scaffold_contract_failed: pipeline not created: {pipeline_path}"
                        + (f" (last_failure: {last_failure_reason})" if last_failure_reason else "")
                        + (f" (root_cause: {root_cause})" if root_cause else "")
                        + (f" (agent_error: {scaffold_err})" if scaffold_err else "")
                    )
            finally:
                if created_agent and agent is not None:
                    try:
                        agent.close()
                    except Exception:
                        pass

            try:
                parsed = load_pipeline_spec(pipeline_path)
            except Exception:
                if not provenance_written:
                    try:
                        report = build_contract_provenance_report(
                            repo=repo_root, purpose="scaffold_contract",
                            strict_opencode=(not bool(seed_stage_skeleton) and not bool(write_fallback_pipeline_yml)),
                            before=contract_before, after=snapshot_contract_files(repo_root),
                            tool_trace=tool_trace, runner_written_paths=runner_written_paths,
                        )
                        dump_provenance(out_dir / "scaffold_provenance.json", report)
                        provenance_written = True
                    except Exception:
                        pass
                raise
            report = validate_scaffold_contract(
                repo_root, pipeline=parsed, require_metrics=bool(scaffold_require_metrics),
            )
            if report.errors:
                if not provenance_written:
                    try:
                        report2 = build_contract_provenance_report(
                            repo=repo_root, purpose="scaffold_contract",
                            strict_opencode=(not bool(seed_stage_skeleton) and not bool(write_fallback_pipeline_yml)),
                            before=contract_before, after=snapshot_contract_files(repo_root),
                            tool_trace=tool_trace, runner_written_paths=runner_written_paths,
                        )
                        dump_provenance(out_dir / "scaffold_provenance.json", report2)
                        provenance_written = True
                    except Exception:
                        pass
                write_text(
                    out_dir / "scaffold_error.txt",
                    "scaffold_contract_failed: incomplete_contract\n"
                    + "\n".join([f"- {x}" for x in report.errors])
                    + ("\n" if report.errors else "")
                    + (
                        "Non-fatal warnings:\n"
                        + "\n".join([f"- {x}" for x in (report.warnings or [])])
                        + ("\n" if report.warnings else "")
                        if report.warnings
                        else ""
                    ),
                )
                raise RuntimeError(
                    f"scaffold_contract_failed: incomplete contract: {report.errors}"
                    + (f" (agent_error: {scaffold_err})" if scaffold_err else "")
                )
            if report.warnings:
                write_text(
                    out_dir / "scaffold_validation_warning.txt",
                    "\n".join([f"- {x}" for x in report.warnings]) + "\n",
                )
            if not provenance_written:
                try:
                    report3 = build_contract_provenance_report(
                        repo=repo_root, purpose="scaffold_contract",
                        strict_opencode=(not bool(seed_stage_skeleton) and not bool(write_fallback_pipeline_yml)),
                        before=contract_before, after=snapshot_contract_files(repo_root),
                        tool_trace=tool_trace, runner_written_paths=runner_written_paths,
                    )
                    dump_provenance(out_dir / "scaffold_provenance.json", report3)
                    provenance_written = True
                except Exception:
                    pass
        elif require_pipeline:
            raise FileNotFoundError(f"pipeline not found: {pipeline_path}")
        else:
            pipeline = PipelineSpec()
            return EnvHandle(repo=repo_root, pipeline_path=pipeline_path, pipeline=pipeline)
    pipeline = load_pipeline_spec(pipeline_path)
    return EnvHandle(repo=repo_root, pipeline_path=pipeline_path, pipeline=pipeline)
