from __future__ import annotations

import json
import os
from pathlib import Path

from .provenance import build_contract_provenance_report, dump_provenance, snapshot_contract_files
from ..opencode.client import OpenCodeClient
from ..core.pipeline_spec import load_pipeline_spec
from .validation import validate_scaffold_contract
from ..utils.subprocess import tail

def _read_text_tail(path: Path, *, n: int) -> str:
    try:
        return tail(path.read_text(encoding="utf-8", errors="replace"), n)
    except Exception:
        return ""

def _bootstrap_artifacts_tail(deploy_artifacts_dir: Path) -> str:
    """Best-effort: extract the most relevant bootstrap failure details from artifacts.

    Note: bootstrap failures happen before deploy_setup.sh runs, so deploy_setup_stderr.txt
    is often empty/missing. Showing the failing bootstrap cmd helps the repair agent
    remove/adjust the problematic bootstrap steps.
    """
    deploy_artifacts_dir = Path(deploy_artifacts_dir).resolve()
    bdir = (deploy_artifacts_dir / "bootstrap").resolve()
    if not bdir.exists():
        return ""

    chunks: list[str] = []
    summary = _read_text_tail(bdir / "bootstrap_summary.json", n=2000).strip()
    if summary:
        chunks.append("bootstrap_summary.json:\n" + summary)

    stderr_paths: list[Path] = []
    try:
        stderr_paths = [p for p in bdir.glob("bootstrap_cmd*_try*_stderr.txt") if p.is_file()]
    except Exception:
        stderr_paths = []

    if stderr_paths:
        keyed: list[tuple[float, str, Path]] = []
        for p in stderr_paths:
            try:
                mt = float(p.stat().st_mtime)
            except Exception:
                mt = 0.0
            keyed.append((mt, p.name, p))
        keyed.sort(key=lambda t: (t[0], t[1]), reverse=True)
        newest = keyed[0][2]
        base = newest.name[: -len("_stderr.txt")] if newest.name.endswith("_stderr.txt") else newest.name
        cmd_tail = _read_text_tail(bdir / f"{base}_cmd.txt", n=2000).strip()
        out_tail = _read_text_tail(bdir / f"{base}_stdout.txt", n=4000).strip()
        err_tail = _read_text_tail(bdir / f"{base}_stderr.txt", n=4000).strip()
        if cmd_tail:
            chunks.append("failing_bootstrap_cmd:\n" + cmd_tail)
        if out_tail:
            chunks.append("failing_bootstrap_stdout_tail:\n" + out_tail)
        if err_tail:
            chunks.append("failing_bootstrap_stderr_tail:\n" + err_tail)

    return ("\n\n".join(chunks) + "\n") if chunks else ""

def repair_contract(
    *,
    repo: Path,
    model: str,
    opencode_url: str,
    unattended: str,
    artifacts_dir: Path,
    failed_stage: str,
    deploy_artifacts_dir: Path,
    rollout_eval_artifacts_dir: Path,
    llm_kind: str | None = None,
    llm_model: str | None = None,
    command_hints: list[str] | None = None,
    extra_context: str = "",
    timeout_seconds: int = 300,
    retry_attempts: int = 2,
    retry_backoff_seconds: float = 2.0,
    context_length: int | None = None,
    max_prompt_chars: int | None = None,
    auto_compact: bool | None = None,
) -> None:
    """Ask OpenCode to repair the repo-local contract under `.opencode_fsm/` (best-effort).

    Hard constraints are enforced in the prompt:
    - may ONLY write `.opencode_fsm/**`
    - may NOT modify `pipeline.yml`
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    oc_username = None
    oc_password = None
    if str(opencode_url or "").strip():
        oc_username = str(os.environ.get("OPENCODE_SERVER_USERNAME") or "opencode").strip() or "opencode"
        oc_password = str(os.environ.get("OPENCODE_SERVER_PASSWORD") or "").strip() or None

    agent = OpenCodeClient(
        repo=repo,
        plan_rel="PLAN.md",
        pipeline_rel="pipeline.yml",
        model=str(model or "").strip(),
        base_url=(str(opencode_url or "").strip() or None),
        timeout_seconds=int(timeout_seconds or 300),
        request_retry_attempts=int(retry_attempts or 0),
        request_retry_backoff_seconds=float(retry_backoff_seconds or 0.0),
        context_length=(int(context_length) if context_length is not None else None),
        max_prompt_chars=(int(max_prompt_chars) if max_prompt_chars is not None else None),
        bash_mode="restricted",
        scaffold_bash_mode="full",
        unattended=str(unattended or "strict"),
        auto_compact=auto_compact,
        server_log_path=artifacts_dir / "opencode_server.log",
        session_title=f"{repo.name}:repair",
        username=oc_username,
        password=oc_password,
        permission_overrides={"external_directory": {"*": "allow"}},
    )
    contract_before = snapshot_contract_files(repo)
    tool_trace: list[dict[str, object]] | None = None
    try:
        deploy_setup_out = _read_text_tail(deploy_artifacts_dir / "deploy_setup_stdout.txt", n=4000)
        deploy_setup_err = _read_text_tail(deploy_artifacts_dir / "deploy_setup_stderr.txt", n=4000)
        deploy_health_out = _read_text_tail(deploy_artifacts_dir / "deploy_health_stdout.txt", n=4000)
        deploy_health_err = _read_text_tail(deploy_artifacts_dir / "deploy_health_stderr.txt", n=4000)
        bootstrap_artifacts = _bootstrap_artifacts_tail(deploy_artifacts_dir)
        rollout_err = _read_text_tail(rollout_eval_artifacts_dir / "rollout_stderr.txt", n=4000)
        eval_err = _read_text_tail(rollout_eval_artifacts_dir / "evaluation_stderr.txt", n=4000)
        repo2 = Path(repo).resolve()
        pipeline_path = (repo2 / "pipeline.yml").resolve()
        if not pipeline_path.exists():
            contract_validation = "pipeline.yml_missing"
        else:
            try:
                pipeline = load_pipeline_spec(pipeline_path)
            except Exception as e:
                contract_validation = f"pipeline_parse_error: {e}"
            else:
                try:
                    report = validate_scaffold_contract(repo2, pipeline=pipeline, require_metrics=True)
                except Exception as e:
                    contract_validation = f"contract_validation_failed: {e}"
                else:
                    payload = {
                        "ok": bool(not report.errors),
                        "errors": list(report.errors),
                        "warnings": list(report.warnings or []),
                    }
                    try:
                        text = json.dumps(payload, ensure_ascii=False, indent=2)
                    except Exception:
                        text = str(payload)
                    contract_validation = tail(text, 8000)
        # When evaluation uses `runner.generic_evaluation`, the real failure reason is typically
        # recorded in `.opencode_fsm/hints_run.json` rather than evaluation_stderr.txt.
        hints_run_preview = _read_text_tail(repo / ".opencode_fsm" / "hints_run.json", n=6000)
        hints_used_preview = _read_text_tail(repo / ".opencode_fsm" / "hints_used.json", n=3000)
        bootstrap_preview = _read_text_tail(repo / ".opencode_fsm" / "bootstrap.yml", n=3000)
        deploy_setup_sh = _read_text_tail(repo / ".opencode_fsm" / "stages" / "deploy_setup.sh", n=4000)
        deploy_health_sh = _read_text_tail(repo / ".opencode_fsm" / "stages" / "deploy_health.sh", n=2000)
        deploy_teardown_sh = _read_text_tail(repo / ".opencode_fsm" / "stages" / "deploy_teardown.sh", n=2000)
        rollout_sh = _read_text_tail(repo / ".opencode_fsm" / "stages" / "rollout.sh", n=4000)
        evaluation_sh = _read_text_tail(repo / ".opencode_fsm" / "stages" / "evaluation.sh", n=4000)
        runtime_env_preview = _read_text_tail(repo / ".opencode_fsm" / "runtime_env.json", n=2000)
        metrics_preview = _read_text_tail(repo / ".opencode_fsm" / "metrics.json", n=2000)
        extra = str(extra_context or "").strip()
        extra_block = f"\n[EXTRA_CONTEXT]\n{extra}\n" if extra else ""
        hints = [str(s).strip() for s in (command_hints or []) if str(s).strip()]
        hints_block = ""
        if hints:
            shown = hints[:20]
            hints_block = (
                "\n"
                "[CANDIDATE_COMMAND_HINTS]\n"
                "These commands were extracted from the repo docs (README/docs). Prefer using them for real execution.\n"
                "If hints exist, evaluation.sh MUST run at least one hint (or a direct adaptation) and derive score from its outputs.\n"
                "Do NOT replace repo-provided evaluation with proxy scoring (micro-benchmarks). If hints cannot run, write ok=false + reason and EXIT NON-ZERO.\n"
                + "".join([f"- {line}\n" for line in shown])
                + ("\n" if len(hints) > len(shown) else "")
            )
        prompt = (
            "You are a contract repair agent.\n"
            "\n"
            "Goal: fix the repo-local contract so the runner can execute deploy -> rollout -> evaluation without errors.\n"
            "\n"
            "Tool usage (REQUIRED):\n"
            "- You MUST implement changes by emitting REAL OpenCode XML tool calls. Do NOT just describe edits.\n"
            "- Do NOT wrap tool calls in markdown fences.\n"
            "- Supported formats:\n"
            "  - Read:  <read filePath=\"...\" />\n"
            "  - Write: <write filePath=\"...\">...content...</write>\n"
            "  - Edit:  <edit filePath=\"...\" oldString=\"...\" newString=\"...\" />\n"
            "  - Bash:  <bash command=\"...\" description=\"...\" />\n"
            "\n"
            "Hard constraints:\n"
            "1) You may ONLY write files under `.opencode_fsm/`.\n"
            "2) Do NOT modify `pipeline.yml`.\n"
            "3) Keep everything non-interactive (assume unattended strict).\n"
            "4) Do NOT use `sed -i` (not portable).\n"
            "5) If you embed python via heredoc, do NOT rely on sys.argv and NEVER put shell args after the heredoc terminator.\n"
            "6) If you need Python, you MUST use `$OPENCODE_FSM_PYTHON` (preferred) or `python3`. Do NOT call `python`.\n"
            "7) Do NOT use `&>/dev/null` (it is prone to escaping/corruption). Use `>/dev/null 2>&1` or `2>/dev/null`.\n"
            "8) Do NOT delete `.opencode_fsm/artifacts/**` (keep evidence).\n"
            "   Avoid deleting/rewriting an existing `.opencode_fsm/venv/**`.\n"
            "   If you need a different Python version (e.g. deps don't have wheels for the current Python),\n"
            "   create a NEW venv directory like `.opencode_fsm/venv_py311` and repoint PATH + OPENCODE_FSM_PYTHON.\n"
            "\n"
            "Contract requirements:\n"
            "- deploy must write `.opencode_fsm/runtime_env.json` (JSON object)\n"
            "- rollout must write `.opencode_fsm/rollout.json` (JSON object)\n"
            "- evaluation must write `.opencode_fsm/metrics.json` (JSON object, must contain keys `ok` and `score`; require `ok=true` for success)\n"
            "  `metrics.json.ok` MUST be a JSON boolean true/false. Do NOT write 1/0 and do NOT write a string.\n"
            "- If command hints exist (see [CANDIDATE_COMMAND_HINTS]), evaluation MUST run at least one hinted/official command.\n"
            "  Additionally, when `OPENCODE_FSM_REQUIRE_HINTS=1`, evaluation MUST write `.opencode_fsm/hints_used.json` with:\n"
            "    - `ok`: boolean (true only if a hinted/official command succeeded)\n"
            "    - `used_anchors`: list[string] (must include at least one token from `OPENCODE_FSM_HINT_ANCHORS_JSON`)\n"
            "    - `commands`: list[string] (recommended; attempted commands)\n"
            "    - `reason`: string (required when ok=false)\n"
            "  If no hinted/official command can run, set ok=false with a clear reason and EXIT NON-ZERO.\n"
            "- Preferred benchmark-agnostic evaluation implementation: call the built-in helper in evaluation.sh:\n"
            "    `$OPENCODE_FSM_PYTHON \"$OPENCODE_FSM_RUNNER_ROOT/runner/generic_evaluation.py\"`\n"
            "  IMPORTANT: this helper script does NOT take CLI flags; it reads inputs from env vars only. Do NOT pass `--runtime-env`/`--metrics-json` etc.\n"
            "  IMPORTANT: the helper scripts live under `$OPENCODE_FSM_RUNNER_ROOT/runner/` (do NOT reference `$OPENCODE_FSM_RUNNER_ROOT/.opencode_fsm/...`).\n"
            "  It already executes `OPENCODE_FSM_HINTS_JSON` safely, writes `hints_used.json` + `metrics.json`, and exits non-zero when required hints fail.\n"
            "  Do NOT hand-roll JSON parsing of hint env vars (easy to get wrong).\n"
            "  CRITICAL: do NOT generate `.opencode_fsm/hints_used.json` manually (no hardcoded anchors, no copying anchors from env).\n"
            "  If you see failures like `hints_used.no_expected_anchor` or audit errors like\n"
            "  `evaluation.sh does not appear to use any doc-derived benchmark/eval command hint anchors`,\n"
            "  the fix is to call the helper above so the runner can verify real hint execution.\n"
            "- If hinted/official commands fail due to missing deps, import errors, or version mismatches, DO NOT fake success.\n"
            "  Instead, add `.opencode_fsm/bootstrap.yml` to set up an isolated environment (e.g., venv) and install repo deps deterministically,\n"
            "  then ensure evaluation runs the hinted command within that environment.\n"
            "  IMPORTANT: bootstrap.yml is for **environment preparation only**. Do NOT run evaluation/test/benchmark commands in bootstrap\n"
            "  (e.g. do NOT run `pytest`, benchmark CLI entrypoints, or `make test`). Those belong in the stage scripts (especially evaluation.sh).\n"
            "  IMPORTANT: if you create a venv, all installs MUST use the venv interpreter (`.opencode_fsm/venv/bin/python -m pip ...`),\n"
            "  not the system `python3 -m pip ...` (do not pollute global/user site-packages).\n"
            "  Example bootstrap.yml pattern:\n"
            "    version: 1\n"
            "    cmds:\n"
            "      - uv python install 3.11\n"
            "      - uv venv --python 3.11 .opencode_fsm/venv_py311\n"
            "      - uv pip install --python .opencode_fsm/venv_py311/bin/python -e .\n"
            "    env:\n"
            "      PATH: \".opencode_fsm/venv_py311/bin:$PATH\"\n"
            "      OPENCODE_FSM_PYTHON: \".opencode_fsm/venv_py311/bin/python\"\n"
            "  If (and only if) you see a tree-sitter language/version mismatch error,\n"
            "  fix it in bootstrap.yml by pinning compatible versions in the venv before running the hinted command.\n"
            "  Prefer versions that have wheels for your Python version; avoid source builds when possible.\n"
            "  If `pip install ...` fails with `No matching distribution found`, do NOT keep guessing random package names/versions.\n"
            "  Either remove the unnecessary dependency, install without pinning, or pick a version that actually exists (pip often prints a list).\n"
            "- NEVER set `ok=true` unless the evaluation actually ran and succeeded.\n"
            "- NEVER hardcode a non-zero score. Derive the score from real execution outputs.\n"
            "- NOTE: the runner audits `.opencode_fsm/stages/evaluation.sh` and will reject hardcoded non-zero scores and proxy/no-op evaluations.\n"
            "- rollout MUST also write a samples JSONL file under `$OPENCODE_FSM_ARTIFACTS_DIR` and include its path in `rollout.json.paths.samples_jsonl`.\n"
            "  Each JSONL line must be an object with keys: `prompt` (string), `completion` (string), `reward` (number).\n"
            "  At least one sample line MUST have a non-empty `completion` (non-whitespace), or the rollout contract will fail.\n"
            "- IMPORTANT: if [EXTRA_CONTEXT] mentions `hf_qa_samples_too_few` OR `hf_qa_prompts_not_anchored`, this target is an HF QA snapshot.\n"
            "  Rollout MUST:\n"
            "  - generate at least N valid samples if N is specified (and have prompt diversity)\n"
            "  - anchor prompts to the HF test parquet questions (i.e., include the real question text; do NOT synthesize unrelated tasks)\n"
            "  In this case, do NOT hand-roll placeholder sample generators.\n"
            "  Preferred benchmark-agnostic fix: call the built-in helper (recommended): `$OPENCODE_FSM_PYTHON \"$OPENCODE_FSM_RUNNER_ROOT/runner/generic_rollout.py\"`.\n"
            "  IMPORTANT: this helper script does NOT take CLI flags; it reads inputs from env vars only.\n"
            "  It already handles HF QA snapshots and respects `OPENCODE_EVAL_MODE` / `OPENCODE_EVAL_LIMIT`.\n"
            "- If a hinted/official command fails due to permission errors writing outputs (e.g., cannot create a results directory),\n"
            "  redirect its outputs to a writable location under `$OPENCODE_FSM_ARTIFACTS_DIR` (or `.opencode_fsm/`).\n"
            "  Prefer passing an explicit output flag (common names: `--output`, `--output_file`, `--out`, `--out_dir`, `--root`) rather than relying on tool defaults.\n"
            "- scripts MUST support `OPENCODE_RUNTIME_ENV_PATH` and local/remote inference inputs (`OPENCODE_TRAINED_MODEL_DIR` OR `OPENCODE_LLM_KIND=remote` + `OPENCODE_LLM_MODEL`, with endpoint/auth from env like `OPENAI_API_KEY`).\n"
            "- If you use an OpenAI-compatible client library that requires an API key, DO NOT embed secrets in files. Instead, read it from env (e.g. OPENAI_API_KEY) and fail clearly if missing.\n"
            "- deploy_teardown should stop any started services/containers (best-effort)\n"
            "\n"
            "[LLM_RUNTIME]\n"
            f"OPENCODE_LLM_KIND: {str(llm_kind or '')}\n"
            f"OPENCODE_LLM_MODEL: {str(llm_model or '')}\n"
            "\n"
            "NOTE:\n"
            "- If `OPENCODE_LLM_KIND=remote`, deploy MUST NOT require a local server. It should write runtime_env.json with `inference.type=openai_compat`\n"
            "  and pass health checks without server.pid.\n"
            "- If `OPENCODE_LLM_KIND=local_hf`, deploy may start a local OpenAI-compatible server and record its base_url + pid.\n"
            "\n"
            f"FAILED_STAGE: {failed_stage}\n"
            f"REPO_ROOT: {repo}\n"
            f"DEPLOY_ARTIFACTS_DIR: {deploy_artifacts_dir}\n"
            f"ROLLOUT_EVAL_ARTIFACTS_DIR: {rollout_eval_artifacts_dir}\n"
            f"{hints_block}"
            "\n"
            "[CONTRACT_VALIDATION_SNAPSHOT]\n"
            f"{contract_validation}\n"
            "\n"
            "[DEPLOY_SETUP_STDERR_TAIL]\n"
            f"{deploy_setup_err}\n"
            "\n"
            "[DEPLOY_SETUP_STDOUT_TAIL]\n"
            f"{deploy_setup_out}\n"
            "\n"
            "[DEPLOY_HEALTH_STDERR_TAIL]\n"
            f"{deploy_health_err}\n"
            "\n"
            "[DEPLOY_HEALTH_STDOUT_TAIL]\n"
            f"{deploy_health_out}\n"
            "\n"
            "[BOOTSTRAP_ARTIFACTS_TAIL]\n"
            f"{bootstrap_artifacts}\n"
            "\n"
            "[ROLLOUT_STDERR_TAIL]\n"
            f"{rollout_err}\n"
            "\n"
            "[EVALUATION_STDERR_TAIL]\n"
            f"{eval_err}\n"
            "\n"
            "[HINTS_RUN_JSON_TAIL]\n"
            f"{hints_run_preview}\n"
            "\n"
            "[HINTS_USED_JSON_TAIL]\n"
            f"{hints_used_preview}\n"
            "\n"
            "[BOOTSTRAP_YML_TAIL]\n"
            f"{bootstrap_preview}\n"
            "\n"
            "[DEPLOY_SETUP_SH_TAIL]\n"
            f"{deploy_setup_sh}\n"
            "\n"
            "[DEPLOY_HEALTH_SH_TAIL]\n"
            f"{deploy_health_sh}\n"
            "\n"
            "[DEPLOY_TEARDOWN_SH_TAIL]\n"
            f"{deploy_teardown_sh}\n"
            "\n"
            "[ROLLOUT_SH_TAIL]\n"
            f"{rollout_sh}\n"
            "\n"
            "[EVALUATION_SH_TAIL]\n"
            f"{evaluation_sh}\n"
            "\n"
            "[RUNTIME_ENV_JSON_TAIL]\n"
            f"{runtime_env_preview}\n"
            "\n"
            "[METRICS_JSON_TAIL]\n"
            f"{metrics_preview}\n"
            f"{extra_block}"
            "\n"
            "Now: inspect `.opencode_fsm/stages/*.sh` and fix the scripts so the contract passes.\n"
        )
        try:
            res = agent.run(prompt, fsm_state="S_REPAIR", iter_idx=0, purpose="repair_contract")
            tool_trace = [dict(x) for x in (res.tool_trace or []) if isinstance(x, dict)]
            # The agent may mutate `.opencode_fsm/**` (including artifacts); keep runner reporting best-effort.
            try:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                (artifacts_dir / "repair_agent_result.txt").write_text(
                    tail(res.assistant_text or "", 20000) + "\n", encoding="utf-8"
                )
            except Exception:
                pass
        except Exception as e:
            # A timeout/error might still have produced partial file writes.
            try:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                (artifacts_dir / "repair_agent_error.txt").write_text(tail(str(e), 4000) + "\n", encoding="utf-8")
            except Exception:
                pass
            return
    finally:
        try:
            try:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            report = build_contract_provenance_report(
                repo=repo,
                purpose="repair_contract",
                strict_opencode=True,
                before=contract_before,
                after=snapshot_contract_files(repo),
                tool_trace=tool_trace,
                runner_written_paths=set(),
            )
            dump_provenance(artifacts_dir / "repair_provenance.json", report)
        except Exception:
            pass
        try:
            agent.close()
        except Exception:
            pass
