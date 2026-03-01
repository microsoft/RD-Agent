from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._util import _ensure_openai_v1_base, _find_hf_test_parquet, _read_json_object
from .contract.hints import suggest_contract_hints
from .contract.repair import repair_contract
from .utils.eval_audit import (
    audit_eval_script_for_hardcoded_nonzero_score,
    audit_eval_script_has_real_execution,
    audit_eval_script_mentions_any_anchor,
)
from .core import (
    DeployCallResult,
    EnvHandle,
    EvaluationCallResult,
    RolloutCallResult,
    deploy as _deploy,
    deploy_teardown as _deploy_teardown,
    evaluate as _evaluate,
    open_env,
    rollout as _rollout,
    rollout_and_evaluate as _rollout_and_evaluate,
    with_runtime_env_path,
)
from .core.stage_cache import invalidate_all_caches

__all__ = ["EnvSession", "setup"]


def _hf_parquet_qa_rows(repo_root: Path) -> int | None:
    """If repo_root is an HF snapshot with a QA test parquet, return its row count."""
    repo_root = Path(repo_root).resolve()
    if not (repo_root / "data" / "hf_manifest.json").exists():
        return None
    parquet_path = _find_hf_test_parquet(repo_root)
    if parquet_path is None:
        return None
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return None
    try:
        pf = pq.ParquetFile(parquet_path)
    except Exception:
        return None
    try:
        schema_names = set(str(n) for n in (pf.schema.names or []))
    except Exception:
        schema_names = set()
    if not {"question", "answer"}.issubset(schema_names):
        return None
    try:
        meta = pf.metadata
        if meta is not None:
            n = int(meta.num_rows)
            return n if n > 0 else None
    except Exception:
        return None
    return None


def _hf_parquet_qa_question_samples(repo_root: Path, *, max_questions: int = 20) -> list[str] | None:
    """If repo_root is an HF QA snapshot, return up to N sample questions from the test parquet."""
    repo_root = Path(repo_root).resolve()
    if not (repo_root / "data" / "hf_manifest.json").exists():
        return None
    parquet_path = _find_hf_test_parquet(repo_root)
    if parquet_path is None:
        return None
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return None
    try:
        pf = pq.ParquetFile(parquet_path)
    except Exception:
        return None
    try:
        schema_names = set(str(n) for n in (pf.schema.names or []))
    except Exception:
        schema_names = set()
    if "question" not in schema_names:
        return None

    n = max(1, int(max_questions))
    out: list[str] = []
    try:
        for batch in pf.iter_batches(batch_size=n, columns=["question"]):
            try:
                arr = batch.column(0).to_pylist()
            except Exception:
                arr = []
            for q in arr:
                s = str(q or "").strip()
                if s:
                    out.append(s)
            break
    except Exception:
        return None
    return out[:n] if out else None


def _validate_rollout_samples(
    repo: Path,
    rollout_path: Path | None,
    *,
    mode: str,
    eval_limit: int | None,
) -> tuple[bool, str]:
    """Validate that rollout produced a usable samples JSONL reference."""
    repo = Path(repo).resolve()
    p = (rollout_path or (repo / ".opencode_fsm" / "rollout.json")).resolve()
    if not p.exists():
        return False, f"missing_rollout_json: {p}"
    obj = _read_json_object(p)
    if obj is None:
        return False, "rollout_json_not_object"

    counts = obj.get("counts")
    if isinstance(counts, dict):
        raw_samples = counts.get("samples")
        raw_errors = counts.get("errors")
        try:
            n_samples = int(raw_samples) if isinstance(raw_samples, (int, float, str)) else None
        except Exception:
            n_samples = None
        try:
            n_errors = int(raw_errors) if isinstance(raw_errors, (int, float, str)) else None
        except Exception:
            n_errors = None
        if (
            isinstance(n_samples, int)
            and isinstance(n_errors, int)
            and n_samples > 0
            and n_errors > 0
            and n_errors >= n_samples
        ):
            return False, f"rollout_counts_all_errors: errors={n_errors} samples={n_samples}"

    paths = obj.get("paths")
    if not isinstance(paths, dict):
        return False, "rollout_json_missing_paths"
    raw = paths.get("samples_jsonl")
    if not isinstance(raw, str) or not raw.strip():
        return False, "rollout_json_missing_paths.samples_jsonl"
    samples_path = Path(raw.strip())
    if not samples_path.is_absolute():
        samples_path = (repo / samples_path).resolve()
    if not samples_path.exists():
        return False, f"samples_jsonl_not_found: {samples_path}"
    try:
        if samples_path.stat().st_size <= 0:
            return False, "samples_jsonl_empty"
    except Exception:
        pass

    mode2 = str(mode or "").strip().lower() or "smoke"
    default_limit = 64 if mode2 == "full" else 8
    try:
        lim = int(eval_limit) if eval_limit is not None else int(default_limit)
    except Exception:
        lim = int(default_limit)
    lim = max(1, int(lim))

    qa_rows = _hf_parquet_qa_rows(repo)
    expected_min = min(int(qa_rows), int(lim)) if isinstance(qa_rows, int) and qa_rows > 0 else None
    distinct_target = 1 if not expected_min or expected_min <= 1 else min(10, int(expected_min))
    qa_questions = _hf_parquet_qa_question_samples(repo, max_questions=20) if expected_min is not None else None

    qa_q_norms: list[str] = []
    if isinstance(qa_questions, list):
        seen: set[str] = set()
        for q in qa_questions:
            qn = " ".join(str(q or "").strip().lower().split())
            if not qn or qn in seen:
                continue
            seen.add(qn)
            qa_q_norms.append(qn)
    qa_anchor_target = min(5, len(qa_q_norms)) if qa_q_norms else 0
    matched_anchors: set[str] = set()

    valid = 0
    nonempty_completions = 0
    distinct_prompts: set[str] = set()
    to_scan = int(expected_min or 1)
    diversity_scan_cap = max(1, min(200, to_scan))
    try:
        with samples_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    item = json.loads(s)
                except Exception:
                    continue
                if not isinstance(item, dict):
                    continue
                prompt = item.get("prompt")
                completion = item.get("completion")
                reward = item.get("reward")
                if isinstance(prompt, str) and isinstance(completion, str) and isinstance(reward, (int, float)):
                    valid += 1
                    if completion.strip():
                        nonempty_completions += 1
                    if len(distinct_prompts) < diversity_scan_cap:
                        distinct_prompts.add(prompt)
                        if qa_anchor_target > 0 and len(matched_anchors) < qa_anchor_target:
                            pn = " ".join(str(prompt or "").strip().lower().split())
                            for qn in qa_q_norms:
                                if qn and qn in pn:
                                    matched_anchors.add(qn)
                                    break
                    if expected_min is None and nonempty_completions >= 1:
                        return True, "ok"
                    if (
                        expected_min is not None
                        and valid >= int(expected_min)
                        and nonempty_completions >= 1
                        and len(distinct_prompts) >= int(distinct_target)
                        and (qa_anchor_target <= 0 or len(matched_anchors) >= int(qa_anchor_target))
                    ):
                        return True, "ok"
    except Exception as e:
        return False, f"failed_to_read_samples_jsonl: {e}"

    if valid <= 0:
        return False, "samples_jsonl_has_no_valid_samples"
    if nonempty_completions <= 0:
        return False, "samples_jsonl_all_empty_completions"
    if expected_min is not None and valid < int(expected_min):
        return False, f"hf_qa_samples_too_few: expected>={expected_min} got={valid}"
    if expected_min is not None and len(distinct_prompts) < int(distinct_target):
        return False, f"hf_qa_prompts_not_diverse: expected>={distinct_target} got={len(distinct_prompts)}"
    if expected_min is not None and qa_anchor_target > 0 and len(matched_anchors) < int(qa_anchor_target):
        return False, f"hf_qa_prompts_not_anchored: expected>={qa_anchor_target} got={len(matched_anchors)}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Shared helpers for EnvSession (extracted from duplicated rollout/evaluate code)
# ---------------------------------------------------------------------------

def _resolve_llm_arg(session: "EnvSession", llm: str | Path) -> None:
    """Resolve an LLM argument (Path or model ID string) and update session state."""
    if isinstance(llm, Path):
        p = llm.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"llm_path_not_found: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"llm_not_directory: {p}")
        session.llm_kind = "local_hf"
        session.trained_model_dir = p
        session.llm_model = None
        return

    s = str(llm or "").strip()
    if not s:
        raise ValueError("empty_llm")
    if s.startswith(("/", "./", "../", "~")):
        p = Path(s).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"llm_path_not_found: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"llm_not_directory: {p}")
        session.llm_kind = "local_hf"
        session.trained_model_dir = p
        session.llm_model = None
    else:
        p2 = Path(s)
        if p2.exists() and p2.is_dir():
            session.llm_kind = "local_hf"
            session.trained_model_dir = p2.expanduser().resolve()
            session.llm_model = None
        else:
            session.llm_kind = "remote"
            session.trained_model_dir = None
            session.llm_model = s


def _build_overrides(session: "EnvSession", env_overrides: dict[str, str] | None, mode: str) -> dict[str, str]:
    """Build the full env overrides dict from session state (base URL, hints, LLM kind)."""
    overrides = dict(env_overrides or {})

    # OpenAI base URL
    base = (
        str(overrides.get("OPENAI_BASE_URL") or "").strip()
        or str(overrides.get("OPENAI_API_BASE") or "").strip()
        or str(os.environ.get("OPENAI_BASE_URL") or "").strip()
        or str(os.environ.get("OPENAI_API_BASE") or "").strip()
    )
    if base:
        base = _ensure_openai_v1_base(base)
        overrides.setdefault("OPENAI_BASE_URL", base)
        overrides.setdefault("OPENAI_API_BASE", base)

    overrides.setdefault("OPENCODE_FSM_RUN_ID", str(session.run_id))
    overrides.setdefault("OPENCODE_EVAL_MODE", str(mode or "smoke").strip() or "smoke")

    # Command hints
    if session.command_hints:
        overrides.setdefault("OPENCODE_FSM_REQUIRE_HINTS", "1")
        try:
            overrides.setdefault("OPENCODE_FSM_HINTS_JSON", json.dumps(list(session.command_hints[:20]), ensure_ascii=False))
        except Exception:
            overrides.setdefault("OPENCODE_FSM_HINTS_JSON", "[]")
        try:
            overrides.setdefault("OPENCODE_FSM_HINT_ANCHORS_JSON", json.dumps(list(session.hint_anchors[:20]), ensure_ascii=False))
        except Exception:
            overrides.setdefault("OPENCODE_FSM_HINT_ANCHORS_JSON", "[]")

    # LLM kind / model
    if session.llm_kind:
        overrides.setdefault("OPENCODE_LLM_KIND", str(session.llm_kind))
    if session.llm_kind == "remote" and session.llm_model:
        overrides.setdefault("OPENCODE_LLM_MODEL", str(session.llm_model))
        overrides.setdefault("OPENAI_MODEL", str(session.llm_model))
    if session.trained_model_dir is not None:
        overrides.setdefault("OPENCODE_TRAINED_MODEL_DIR", str(session.trained_model_dir))
    if session.runtime_env_path is not None:
        overrides.update(with_runtime_env_path(session.runtime_env_path))

    kind2 = str(session.llm_kind or "").strip() or "local_hf"
    if kind2 == "remote":
        if not session.llm_model:
            raise ValueError("missing_llm: call deploy/rollout with llm=model_id first")
        overrides["OPENCODE_LLM_KIND"] = "remote"
        overrides["OPENCODE_LLM_MODEL"] = str(session.llm_model)
        overrides.setdefault("OPENAI_MODEL", str(session.llm_model))
        overrides.pop("OPENCODE_TRAINED_MODEL_DIR", None)
    else:
        if session.trained_model_dir is None:
            raise ValueError("missing_llm: call deploy/rollout with llm=model_dir first")
        overrides["OPENCODE_LLM_KIND"] = "local_hf"
        overrides["OPENCODE_TRAINED_MODEL_DIR"] = str(session.trained_model_dir)
        model_id = str(session.trained_model_dir.name or "").strip()
        if model_id:
            overrides.setdefault("OPENAI_MODEL", model_id)
        overrides.pop("OPENCODE_LLM_MODEL", None)

    return overrides


def _apply_runtime_env(session: "EnvSession", overrides: dict[str, str]) -> None:
    """Read runtime_env.json and update overrides with inference endpoint info."""
    if session.runtime_env_path is None:
        return
    overrides.update(with_runtime_env_path(session.runtime_env_path))
    p2 = Path(session.runtime_env_path).expanduser().resolve()
    obj2 = _read_json_object(p2)
    if obj2 is None:
        return

    base_raw = ""
    model_raw = ""
    inf = obj2.get("inference")
    if isinstance(inf, dict):
        base_raw = str(inf.get("openai_base_url") or inf.get("base_url") or "").strip()
        model_raw = str(inf.get("model") or "").strip()
    if not base_raw:
        svc = obj2.get("service")
        if isinstance(svc, dict):
            base_raw = str(svc.get("base_url") or "").strip()
    base_v1 = _ensure_openai_v1_base(base_raw) if base_raw else ""
    base2 = base_v1 or None
    model2 = model_raw or None

    if base2:
        explicit = any(str(overrides.get(k) or "").strip() for k in ("OPENAI_API_BASE", "OPENAI_BASE_URL"))
        if session.llm_kind == "local_hf" or not explicit:
            overrides["OPENAI_API_BASE"] = base2
            overrides["OPENAI_BASE_URL"] = base2
    if model2:
        overrides.setdefault("OPENAI_MODEL", model2)
        overrides.setdefault("OPENCODE_LLM_MODEL", model2)
    if session.llm_kind == "local_hf":
        overrides.setdefault("OPENAI_API_KEY", str(overrides.get("OPENAI_API_KEY") or "local"))


def _extract_verify_errors(eval_res: EvaluationCallResult) -> str:
    """Extract combined error string from a failed evaluation result's verification."""
    verify = eval_res.verify
    if verify is None:
        return ""
    parts: list[str] = []
    failed_stage = getattr(verify, "failed_stage", None)
    if isinstance(failed_stage, str) and failed_stage.strip():
        parts.append(f"verify.failed_stage: {failed_stage.strip()}")
    metrics_errors = getattr(verify, "metrics_errors", None)
    if isinstance(metrics_errors, list):
        cleaned = [str(x).strip() for x in metrics_errors if str(x).strip()]
        if cleaned:
            parts.append("verify.metrics_errors:\n" + "\n".join([f"- {x}" for x in cleaned]))
    return "\n".join(parts).strip()


def _run_eval_audit(session: "EnvSession", eval_res: EvaluationCallResult, audit_mode: str, error_dir: Path) -> str:
    """Run evaluation audits and return combined error string (empty if passed)."""
    if not eval_res.ok:
        return ""

    combined = ""
    if (
        bool(session.require_metrics)
        and isinstance(eval_res.metrics, dict)
        and eval_res.metrics.get("ok") is not True
    ):
        combined = "metrics.ok_not_true"
        try:
            (error_dir / "metrics_contract_error.txt").write_text("metrics.ok_not_true\n", encoding="utf-8")
        except Exception:
            pass
    elif bool(session.require_metrics) and audit_mode != "off":
        audit_issue = audit_eval_script_for_hardcoded_nonzero_score(session.env.repo)
        audit_issue2 = audit_eval_script_has_real_execution(session.env.repo, extra_markers=session.hint_anchors)
        audit_issue3 = audit_eval_script_mentions_any_anchor(session.env.repo, session.hint_anchors)
        combined = "\n\n".join([x for x in (audit_issue, audit_issue2, audit_issue3) if x]).strip()
        if combined:
            try:
                (error_dir / "evaluation_audit_error.txt").write_text(combined + "\n", encoding="utf-8")
            except Exception:
                pass
    return combined


def _call_repair(session: "EnvSession", run_root: Path, attempt: int, failed_stage: str,
                 deploy_dir: Path, rollout_eval_dir: Path, extra_context: str) -> None:
    """Call repair_contract with standard session parameters."""
    invalidate_all_caches(session.env.repo)
    repair_contract(
        repo=session.env.repo,
        model=str(session.opencode_repair_model or session.opencode_model or "").strip(),
        opencode_url=str(session.opencode_url or ""),
        unattended=str(session.unattended or "strict"),
        artifacts_dir=(run_root / f"repair_{attempt+1:02d}").resolve(),
        failed_stage=failed_stage,
        deploy_artifacts_dir=deploy_dir,
        rollout_eval_artifacts_dir=rollout_eval_dir,
        llm_kind=str(session.llm_kind or ""),
        llm_model=str(session.llm_model or ""),
        command_hints=session.command_hints,
        extra_context=extra_context,
        timeout_seconds=int(session.opencode_timeout_seconds or 300),
        retry_attempts=int(session.opencode_retry_attempts or 0),
        retry_backoff_seconds=float(session.opencode_retry_backoff_seconds or 0.0),
        context_length=(int(session.opencode_context_length) if session.opencode_context_length is not None else None),
        max_prompt_chars=(int(session.opencode_max_prompt_chars) if session.opencode_max_prompt_chars is not None else None),
        auto_compact=session.opencode_auto_compact,
    )


# ---------------------------------------------------------------------------
# EnvSession
# ---------------------------------------------------------------------------

@dataclass
class EnvSession:
    """Programmatic wrapper: `setup()` -> `sess.rollout(llm=...)` -> `sess.evaluate()`."""

    env: EnvHandle
    run_id: str
    unattended: str
    opencode_model: str
    opencode_repair_model: str
    opencode_url: str
    opencode_timeout_seconds: int
    require_metrics: bool
    command_hints: list[str]
    hint_anchors: list[str]
    opencode_retry_attempts: int = 2
    opencode_retry_backoff_seconds: float = 2.0
    opencode_context_length: int | None = None
    opencode_max_prompt_chars: int | None = None
    opencode_auto_compact: bool | None = None
    audit: str = "on"
    use_cache: bool = True
    runtime_env_path: Path | None = None
    llm_kind: str = ""
    llm_model: str | None = None
    trained_model_dir: Path | None = None

    def rollout(
        self,
        llm: str | Path,
        *,
        mode: str = "smoke",
        require_samples: bool = False,
        env_overrides: dict[str, str] | None = None,
        artifacts_dir: Path | None = None,
        repair_iters: int = 3,
    ) -> RolloutCallResult:
        """Execute deploy -> rollout with automatic repair on failure."""
        _resolve_llm_arg(self, llm)

        if artifacts_dir is not None:
            run_root = Path(str(artifacts_dir)).expanduser().resolve()
        else:
            run_root = (self.env.repo / ".opencode_fsm" / "artifacts" / str(self.run_id) / "env_api").resolve()
        run_root.mkdir(parents=True, exist_ok=True)

        for attempt in range(int(max(0, repair_iters)) + 1):
            deploy_dir = (run_root / f"deploy_attempt_{attempt+1:02d}").resolve()
            rollout_dir = (run_root / f"rollout_attempt_{attempt+1:02d}").resolve()
            contract_err = ""
            overrides = _build_overrides(self, env_overrides, mode)

            deploy_res = _deploy(
                self.env, artifacts_dir=deploy_dir, env_overrides=overrides,
                unattended=str(self.unattended or "strict"), run_bootstrap_first=True,
                use_cache=self.use_cache,
            )
            if not deploy_res.ok:
                if attempt >= int(max(0, repair_iters)):
                    return RolloutCallResult(ok=False, artifacts_dir=rollout_dir, rollout_path=None, verify=deploy_res.verify)
                _call_repair(self, run_root, attempt,
                             str(getattr(deploy_res.verify, "failed_stage", "") or "deploy"),
                             deploy_dir, rollout_dir, "")
                continue

            self.runtime_env_path = deploy_res.runtime_env_path or (self.env.repo / ".opencode_fsm" / "runtime_env.json").resolve()
            _apply_runtime_env(self, overrides)

            rollout_res = _rollout(
                self.env, artifacts_dir=rollout_dir, env_overrides=overrides,
                unattended=str(self.unattended or "strict"), run_bootstrap_first=True,
                use_cache=self.use_cache,
            )
            if rollout_res.ok and bool(require_samples):
                raw_limit = overrides.get("OPENCODE_EVAL_LIMIT")
                try:
                    lim = int(str(raw_limit).strip()) if str(raw_limit or "").strip() else None
                except Exception:
                    lim = None
                ok_samples, reason = _validate_rollout_samples(
                    self.env.repo, rollout_res.rollout_path,
                    mode=str(mode or "smoke"), eval_limit=lim,
                )
                if not ok_samples:
                    try:
                        (rollout_dir / "rollout_contract_error.txt").write_text(str(reason) + "\n", encoding="utf-8")
                    except Exception:
                        pass
                    contract_err = str(reason)
                    rollout_res = RolloutCallResult(
                        ok=False, artifacts_dir=rollout_res.artifacts_dir,
                        rollout_path=rollout_res.rollout_path, verify=rollout_res.verify,
                    )

            if rollout_res.ok:
                return rollout_res
            if attempt >= int(max(0, repair_iters)):
                return rollout_res

            try:
                _deploy_teardown(
                    self.env,
                    artifacts_dir=(run_root / f"teardown_attempt_{attempt+1:02d}" / "deploy_teardown"),
                    env_overrides=overrides, unattended=str(self.unattended or "strict"),
                )
            except Exception:
                pass
            _call_repair(self, run_root, attempt, "rollout", deploy_dir, rollout_dir,
                         contract_err or ("rollout_contract_invalid" if require_samples else ""))

        return rollout_res  # pragma: no cover

    def evaluate(
        self,
        llm: str | Path | None = None,
        *,
        mode: str = "smoke",
        env_overrides: dict[str, str] | None = None,
        artifacts_dir: Path | None = None,
        repair_iters: int = 3,
    ) -> EvaluationCallResult:
        """Execute evaluation (writes metrics.json), with best-effort teardown."""
        if llm is not None:
            _resolve_llm_arg(self, llm)

        if artifacts_dir is not None:
            run_root = Path(str(artifacts_dir)).expanduser().resolve()
        else:
            run_root = (self.env.repo / ".opencode_fsm" / "artifacts" / str(self.run_id) / "env_api").resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        try:
            if not self.llm_kind:
                raise ValueError("missing_llm: call deploy/rollout with llm=model_dir|model_id first")

            audit_mode = str(self.audit or "on").strip().lower() or "on"
            if audit_mode not in ("on", "off", "warn-only"):
                audit_mode = "on"

            for attempt in range(int(max(0, repair_iters)) + 1):
                deploy_dir = (run_root / f"deploy_attempt_{attempt+1:02d}").resolve()
                roll_eval_dir = (run_root / f"rollout_evaluation_attempt_{attempt+1:02d}").resolve()
                eval_dir = (run_root / f"evaluation_attempt_{attempt+1:02d}").resolve()

                overrides = _build_overrides(self, env_overrides, mode)
                combined = ""

                if self.runtime_env_path is not None and attempt == 0:
                    _apply_runtime_env(self, overrides)
                    eval_res = _evaluate(
                        self.env, artifacts_dir=eval_dir, env_overrides=overrides,
                        unattended=str(self.unattended or "strict"), run_bootstrap_first=True,
                        use_cache=self.use_cache,
                    )
                    if not eval_res.ok:
                        combined = _extract_verify_errors(eval_res)
                    if eval_res.ok:
                        combined = _run_eval_audit(self, eval_res, audit_mode, eval_dir)
                        if not combined or audit_mode == "warn-only":
                            return eval_res
                        eval_res = EvaluationCallResult(
                            ok=False, artifacts_dir=eval_res.artifacts_dir,
                            metrics_path=eval_res.metrics_path, metrics=eval_res.metrics,
                            verify=eval_res.verify,
                        )
                else:
                    deploy_res = _deploy(
                        self.env, artifacts_dir=deploy_dir, env_overrides=overrides,
                        unattended=str(self.unattended or "strict"), run_bootstrap_first=True,
                        use_cache=self.use_cache,
                    )
                    if not deploy_res.ok:
                        if attempt >= int(max(0, repair_iters)):
                            return EvaluationCallResult(
                                ok=False, artifacts_dir=roll_eval_dir,
                                metrics_path=None, metrics=None, verify=deploy_res.verify,
                            )
                        _call_repair(self, run_root, attempt,
                                     str(getattr(deploy_res.verify, "failed_stage", "") or "deploy"),
                                     deploy_dir, roll_eval_dir, "")
                        continue

                    self.runtime_env_path = (
                        deploy_res.runtime_env_path or (self.env.repo / ".opencode_fsm" / "runtime_env.json").resolve()
                    )
                    _apply_runtime_env(self, overrides)

                    _rollout_res, eval_res = _rollout_and_evaluate(
                        self.env, artifacts_dir=roll_eval_dir, env_overrides=overrides,
                        unattended=str(self.unattended or "strict"), run_bootstrap_first=True,
                        use_cache=self.use_cache,
                    )
                    if not eval_res.ok:
                        combined = _extract_verify_errors(eval_res)
                    if eval_res.ok:
                        combined = _run_eval_audit(self, eval_res, audit_mode, roll_eval_dir)
                        if not combined or audit_mode == "warn-only":
                            return eval_res
                        eval_res = EvaluationCallResult(
                            ok=False, artifacts_dir=eval_res.artifacts_dir,
                            metrics_path=eval_res.metrics_path, metrics=eval_res.metrics,
                            verify=eval_res.verify,
                        )

                if attempt >= int(max(0, repair_iters)):
                    return eval_res

                try:
                    _deploy_teardown(
                        self.env,
                        artifacts_dir=(run_root / f"teardown_attempt_{attempt+1:02d}" / "deploy_teardown"),
                        env_overrides=overrides, unattended=str(self.unattended or "strict"),
                    )
                except Exception:
                    pass
                _call_repair(self, run_root, attempt, "evaluation",
                             deploy_dir if deploy_dir.exists() else run_root,
                             roll_eval_dir if roll_eval_dir.exists() else eval_dir,
                             combined)

            return eval_res  # pragma: no cover
        finally:
            overrides = _build_overrides(self, env_overrides, mode)
            if self.runtime_env_path is not None:
                overrides.update(with_runtime_env_path(self.runtime_env_path))
            try:
                _deploy_teardown(
                    self.env,
                    artifacts_dir=(run_root / "final_teardown" / "deploy_teardown"),
                    env_overrides=overrides, unattended=str(self.unattended or "strict"),
                )
            except Exception:
                pass


def setup(
    target: str | Path,
    *,
    clones_dir: Path | None = None,
    pipeline_rel: str = "pipeline.yml",
    require_metrics: bool = True,
    audit: str = "on",
    use_cache: bool = True,
    opencode_model: str = "",
    opencode_repair_model: str | None = None,
    opencode_url: str = "",
    unattended: str = "strict",
    opencode_timeout_seconds: int = 300,
    opencode_repair_timeout_seconds: int | None = None,
    opencode_retry_attempts: int = 2,
    opencode_retry_backoff_seconds: float = 2.0,
    opencode_session_recover_attempts: int | None = None,
    opencode_session_recover_backoff_seconds: float | None = None,
    opencode_context_length: int | None = None,
    opencode_max_prompt_chars: int | None = None,
    opencode_bash: str = "restricted",
    scaffold_opencode_bash: str = "full",
    strict_opencode: bool = True,
    artifacts_dir: Path | None = None,
    opencode_auto_compact: bool | None = None,
) -> EnvSession:
    """Open an environment handle for a target repo/url and ensure a runnable contract exists."""
    clones_dir = Path(str(clones_dir)).expanduser().resolve() if clones_dir is not None else None
    artifacts_dir = Path(str(artifacts_dir)).expanduser().resolve() if artifacts_dir is not None else None
    env_handle: EnvHandle = open_env(
        target,
        clones_dir=clones_dir,
        pipeline_rel=str(pipeline_rel or "pipeline.yml").strip() or "pipeline.yml",
        require_pipeline=True,
        scaffold_contract="opencode",
        scaffold_require_metrics=bool(require_metrics),
        model=str(opencode_model or ""),
        opencode_url=str(opencode_url or ""),
        opencode_timeout_seconds=int(opencode_timeout_seconds or 300),
        opencode_retry_attempts=int(opencode_retry_attempts or 0),
        opencode_retry_backoff_seconds=float(opencode_retry_backoff_seconds or 0.0),
        opencode_session_recover_attempts=(
            int(opencode_session_recover_attempts) if opencode_session_recover_attempts is not None else None
        ),
        opencode_session_recover_backoff_seconds=(
            float(opencode_session_recover_backoff_seconds)
            if opencode_session_recover_backoff_seconds is not None
            else None
        ),
        opencode_context_length=(int(opencode_context_length) if opencode_context_length is not None else None),
        opencode_max_prompt_chars=(int(opencode_max_prompt_chars) if opencode_max_prompt_chars is not None else None),
        opencode_bash=str(opencode_bash or "restricted"),
        scaffold_opencode_bash=str(scaffold_opencode_bash or "full"),
        unattended=str(unattended or "strict"),
        artifacts_dir=artifacts_dir,
        seed_stage_skeleton=not bool(strict_opencode),
        write_fallback_pipeline_yml=not bool(strict_opencode),
        opencode_auto_compact=opencode_auto_compact,
    )
    hints = suggest_contract_hints(env_handle.repo)
    return EnvSession(
        env=env_handle,
        run_id=time.strftime("%Y%m%d_%H%M%S", time.localtime()),
        unattended=str(unattended or "strict"),
        opencode_model=str(opencode_model or ""),
        opencode_repair_model=str(opencode_repair_model or opencode_model or ""),
        opencode_url=str(opencode_url or ""),
        opencode_timeout_seconds=int(opencode_repair_timeout_seconds or opencode_timeout_seconds or 300),
        require_metrics=bool(require_metrics),
        command_hints=list(hints.commands or []),
        hint_anchors=list(hints.anchors or []),
        opencode_retry_attempts=int(opencode_retry_attempts or 0),
        opencode_retry_backoff_seconds=float(opencode_retry_backoff_seconds or 0.0),
        opencode_context_length=(int(opencode_context_length) if opencode_context_length is not None else None),
        opencode_max_prompt_chars=(int(opencode_max_prompt_chars) if opencode_max_prompt_chars is not None else None),
        opencode_auto_compact=opencode_auto_compact,
        audit=str(audit or "on"),
        use_cache=bool(use_cache),
    )
