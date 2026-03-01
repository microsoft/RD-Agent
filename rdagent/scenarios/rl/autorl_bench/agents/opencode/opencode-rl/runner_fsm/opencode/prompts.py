from __future__ import annotations

import json
from pathlib import Path

from ..core.pipeline_verify import fmt_stage_tail
from ..utils.subprocess import STDIO_TAIL_CHARS, tail
from ..dtypes import VerificationResult

def make_plan_update_prompt(snapshot_text: str, test_cmd: str, *, extra: str = "") -> str:
    extra = extra.strip()
    extra_block = f"\n[EXTRA]\n{extra}\n" if extra else ""
    return (
        "You are a strict plan editor. Your job: ONLY edit PLAN.md so it becomes machine-parseable and executable.\n"
        "\n"
        "Tooling:\n"
        "- Use REAL OpenCode XML tool calls (do not wrap them in markdown fences).\n"
        "- READ format: `<read filePath=\"PATH\" />`\n"
        "- WRITE format: `<write filePath=\"PATH\">...file content...</write>`\n"
        "- BASH format (self-closing): `<bash command=\"...\" description=\"...\" />`\n"
        "- For bash, use the self-closing form exactly (`/>`). Do NOT use `<bash>...</bash>`.\n"
        "- Do NOT print pseudo tool snippets as plain text.\n"
        "- Do NOT fabricate `<tool_result>` blocks or simulated tool transcripts.\n"
        "Prefer file WRITE tool calls for any edits/creates (do NOT use bash redirections like `>` to write files).\n"
        "\n"
        "Hard constraints:\n"
        "1) You may ONLY modify PLAN.md. Do NOT touch any other file.\n"
        "2) Under `## Next (exactly ONE item)` there must be exactly ONE unchecked item:\n"
        "   `- [ ] (STEP_ID=NNN) ...`\n"
        "3) Each step must be atomic: one edit + one verification.\n"
        "4) Verification is ALWAYS the runner's TEST_CMD. Write steps so that after completing the step, running TEST_CMD will pass.\n"
        f"5) `## Acceptance` MUST include the exact line (verbatim): - [ ] TEST_CMD passes: `{test_cmd}`\n"
        "6) Do NOT put placeholders like `{TEST_CMD}` into PLAN.md; always use the exact TEST_CMD string.\n"
        "7) The `Next` item must be an IMPLEMENTATION step (edits code/scripts/config), not a meta planning step about editing PLAN.md.\n"
        "8) It is OK if the Next step creates new files; do not mark Blocked just to ask to 'add a file to chat'.\n"
        "9) Put uncertainty into `## Notes`. If you need human input, mark the step Blocked and specify what is needed.\n"
        "\n"
        f"TEST_CMD: {test_cmd}\n"
        "\n"
        f"{snapshot_text}"
        f"{extra_block}"
        "\n"
        "Now: edit PLAN.md only.\n"
    )

def make_execute_prompt(snapshot_text: str, step: dict[str, str]) -> str:
    return (
        "You are a strict executor. Your job: implement ONLY the single `Next` step.\n"
        "\n"
        "Tooling:\n"
        "- Use REAL OpenCode XML tool calls (do not wrap them in markdown fences).\n"
        "- READ format: `<read filePath=\"PATH\" />`\n"
        "- WRITE format: `<write filePath=\"PATH\">...file content...</write>`\n"
        "- BASH format (self-closing): `<bash command=\"...\" description=\"...\" />`\n"
        "- For bash, use the self-closing form exactly (`/>`). Do NOT use `<bash>...</bash>`.\n"
        "- Do NOT print pseudo tool snippets as plain text.\n"
        "- Do NOT fabricate `<tool_result>` blocks or simulated tool transcripts.\n"
        "Prefer file WRITE tool calls for any edits/creates (do NOT use bash redirections like `>` to write files).\n"
        "\n"
        "Hard constraints:\n"
        "1) Do ONLY this one thing. No refactors, no extra features.\n"
        "2) Keep changes as small as possible.\n"
        "3) Do NOT modify PLAN.md.\n"
        "4) Do NOT modify the pipeline YAML (human-owned contract; runner will revert edits).\n"
        "5) You MAY create new files if needed; this run is unattended and file-create/add-to-chat prompts are auto-approved.\n"
        "\n"
        f"NEXT_STEP: (STEP_ID={step['id']}) {step['text']}\n"
        "\n"
        f"{snapshot_text}\n"
    )

def make_scaffold_contract_prompt(
    repo: Path,
    *,
    pipeline_rel: str,
    require_metrics: bool,
    command_hints: list[str] | None = None,
) -> str:
    # For contract runs we want to avoid "silent placeholders". Requiring an explicit `ok` flag
    # makes it easier for downstream automation (and humans) to distinguish real scores vs fallbacks.
    required_keys = ["score", "ok"] if require_metrics else []
    required_line = ""
    if required_keys:
        required_line = f"- `.opencode_fsm/metrics.json` must include keys: {required_keys}\n"
    hints = [str(s).strip() for s in (command_hints or []) if str(s).strip()]
    hints_block = ""
    if hints:
        shown = hints[:20]
        hints_block = (
            "\n"
            "[CANDIDATE_COMMAND_HINTS]\n"
            "The following commands were extracted from repo docs (README/docs). Prefer using them for REAL execution.\n"
            "If hints exist, evaluation.sh MUST run (at least) one hint (or a direct adaptation) and derive score from its outputs.\n"
            "Do NOT replace repo-provided evaluation with proxy scoring (micro-benchmarks). If hints cannot run, write ok=false + reason and EXIT NON-ZERO.\n"
            + "".join([f"- {line}\n" for line in shown])
            + ("\n" if len(hints) > len(shown) else "")
        )
    return (
        "You are a contract scaffolder.\n"
        "\n"
        "Goal: make this repo runnable by the OpenCode-FSM runner in one command by creating a minimal, repo-local contract.\n"
        "Your output must be generic: do NOT embed repo-specific hardcoding unless it is explicitly present in the repo.\n"
        "The runner will NOT contain any benchmark-specific adapters. Therefore your contract must self-describe deploy + runtime env + rollout + evaluation.\n"
        "\n"
        "IMPORTANT: `pipeline.yml` MUST follow the runner's v1 schema (top-level keys like: tests, deploy, rollout, benchmark, auth, "
        "evaluation, security, artifacts). Do NOT invent a custom schema (e.g. DO NOT write a top-level `stages:` list).\n"
        "\n"
        "Hard requirements (must satisfy even if the repo provides no docs):\n"
        "1) `deploy.setup` MUST write `.opencode_fsm/runtime_env.json` (a JSON object).\n"
        "2) `rollout` MUST write `.opencode_fsm/rollout.json` (a JSON object).\n"
        "   Additionally, rollout MUST also write a detailed samples file under `$OPENCODE_FSM_ARTIFACTS_DIR` and reference it from rollout.json.\n"
        "   Required: `rollout.json.paths.samples_jsonl` must point to a JSONL file. Each JSONL line must be an object with keys:\n"
        "     - `prompt`: string\n"
        "     - `completion`: string\n"
        "     - `reward`: number\n"
        "   Contract validation note: at least ONE sample line MUST have a non-empty `completion` (not just whitespace).\n"
        "3) `evaluation` MUST write `.opencode_fsm/metrics.json` (a JSON object).\n"
        "   `metrics.json.ok` MUST be a JSON boolean (true/false). Do NOT write 1/0 and do NOT write a string.\n"
        "   If command hints exist (see [CANDIDATE_COMMAND_HINTS]), evaluation MUST run at least one hinted/official command.\n"
        "   Preferred benchmark-agnostic implementation: in evaluation.sh call `$OPENCODE_FSM_PYTHON \"$OPENCODE_FSM_RUNNER_ROOT/runner/generic_evaluation.py\"`.\n"
        "   It already handles hint execution + hints_used.json + metrics.json + correct exit codes.\n"
        "   It MUST also write `.opencode_fsm/hints_used.json` with:\n"
        "     - `ok`: boolean (true only if at least one hinted/official command succeeded)\n"
        "     - `used_anchors`: list[string] (must include at least one token from `OPENCODE_FSM_HINT_ANCHORS_JSON`)\n"
        "     - `commands`: list[string] (recommended; the exact commands attempted)\n"
        "     - `reason`: string (required when ok=false)\n"
        "   If no hinted/official command can run, set ok=false, write a clear reason, and EXIT NON-ZERO.\n"
        "4) Stages must be non-interactive and bounded (timeouts).\n"
        "5) If real deploy/rollout/evaluation cannot be inferred from the repo, write the required JSON file(s) with `ok=false`/`score=0` and a clear `reason`, and EXIT NON-ZERO.\n"
        "   Do NOT pretend success by exiting 0 with placeholder metrics.\n"
        "6) If `.opencode_fsm/stages/*.sh` already exist, prefer EDITING them instead of rewriting from scratch.\n"
        "7) If you create `.opencode_fsm/bootstrap.yml`, keep a strict schema:\n"
        "   top-level mapping with keys from {version, cmds, env, workdir, timeout_seconds, retries}.\n"
        "   `cmds` must be a list of shell command strings. Do NOT invent nested wrappers/schemas.\n"
        "   IMPORTANT: `.opencode_fsm/bootstrap.yml` is YAML (not a shell script). It MUST NOT start with `#!/bin/bash`.\n"
        "   Minimal valid example (edit for the repo):\n"
        "   version: 1\n"
        "   cmds:\n"
        "     - uv python install 3.11\n"
        "     - uv venv --python 3.11 .opencode_fsm/venv\n"
        "     - uv pip install --python .opencode_fsm/venv/bin/python -e \".[testing]\"\n"
        "   env:\n"
        "     PATH: \".opencode_fsm/venv/bin:$PATH\"\n"
        "     OPENCODE_FSM_PYTHON: \".opencode_fsm/venv/bin/python\"\n"
        "   workdir: .\n"
        "   timeout_seconds: 600\n"
        "   retries: 1\n"
        "\n"
        "runtime_env.json schema (recommended minimal):\n"
        "{\n"
        '  "ts": "...",\n'
        '  "run_id": "...",\n'
        '  "service": { "base_url": "...", "health_url": "..." },\n'
        '  "inference": { "type": "local_hf|openai_compat", "model_dir": "...", "openai_base_url": "...", "model": "...", "notes": "do not store secrets here" },\n'
        '  "paths": { "rollout_path": ".opencode_fsm/rollout.json", "metrics_path": ".opencode_fsm/metrics.json" }\n'
        "}\n"
        "\n"
        "Safety constraints for generated stage scripts:\n"
        "- Prefer repo-provided commands (README/Makefile/CI). Avoid guessing ports/flags.\n"
        "- Your rollout/evaluation scripts MUST support these inputs (do NOT hardcode paths):\n"
        "  - `OPENCODE_RUNTIME_ENV_PATH`: path to runtime_env.json (use if set)\n"
        "  - `OPENCODE_TRAINED_MODEL_DIR`: optional path to a trained HF model directory (use if set; else read runtime_env.json.inference.model_dir if present)\n"
        "  - `OPENCODE_LLM_KIND`: `local_hf` or `remote` (when `remote`, DO NOT require a local model dir)\n"
        "  - `OPENCODE_LLM_MODEL`: model id/name for remote OpenAI-compatible inference (pass through; do NOT hardcode)\n"
        "  - For remote inference, read endpoint/auth from env (e.g. `OPENAI_API_KEY`, optional `OPENAI_API_BASE` / `OPENAI_BASE_URL`). Do NOT hardcode.\n"
        "  - When `OPENCODE_LLM_KIND=remote`, do NOT use local inference backends (e.g. vLLM/HF/ollama) and do NOT download local models. Use an OpenAI-compatible backend and rely on env-provided model/base URL.\n"
        "- When `OPENCODE_FSM_REQUIRE_HINTS=1`, evaluation MUST produce `.opencode_fsm/hints_used.json` as described above.\n"
        "  Hint lists are provided via env vars:\n"
        "    - `OPENCODE_FSM_HINTS_JSON`: JSON array of doc-derived hint strings\n"
        "    - `OPENCODE_FSM_HINT_ANCHORS_JSON`: JSON array of high-signal tokens extracted from hints\n"
        "- If you need Python, you MUST use `$OPENCODE_FSM_PYTHON` (preferred) or `python3`. Do NOT call `python`.\n"
        "- Do NOT use the bash redirection shorthand `&>/dev/null` (it is prone to escaping/corruption in XML-like outputs). Use `>/dev/null 2>&1` or `2>/dev/null` instead.\n"
        "- Do NOT write broken redirects like `2&&1` (typo). Always use `2>&1`.\n"
        "- Prefer writing JSON via python instead of bash heredocs. If you use a heredoc, NEVER use a single-quoted delimiter (e.g. `<< 'EOF'`) when you expect `$VARS` or `$(...)` to expand.\n"
        "- If you need a smoke subset, respect optional `OPENCODE_EVAL_MODE=smoke|full` and `OPENCODE_EVAL_LIMIT` env vars.\n"
        "- Do NOT use `sed -i` (not portable between macOS and Linux). Prefer writing JSON via python.\n"
        "- If you embed python via heredoc, do NOT rely on `sys.argv[...]` (use env vars instead) and NEVER put shell args after the heredoc terminator.\n"
        "- If you need to expose the trained model as a service (local_hf), start an OpenAI-compatible server and write its base_url into runtime_env.json (no hardcoded ports).\n"
        "- If you use an OpenAI-compatible client library that requires an API key, DO NOT embed secrets in files. Instead, read it from env (e.g. OPENAI_API_KEY) and fail clearly if missing.\n"
        "- Built-in helper you may use in stage scripts (runner will set `$OPENCODE_FSM_PYTHON`, PYTHONPATH, and `OPENCODE_FSM_RUNNER_ROOT`):\n"
        "  - Generic rollout (writes rollout.json + samples JSONL): `$OPENCODE_FSM_PYTHON \"$OPENCODE_FSM_RUNNER_ROOT/runner/generic_rollout.py\"`\n"
        "  - Generic evaluation (runs doc/CI hints + writes hints_used.json + metrics.json): `$OPENCODE_FSM_PYTHON \"$OPENCODE_FSM_RUNNER_ROOT/runner/generic_evaluation.py\"`\n"
        "  IMPORTANT: these helper scripts DO NOT take CLI flags; they read inputs from env vars only. Do NOT pass `--runtime-env`/`--metrics-json` etc.\n"
        "  IMPORTANT: do NOT reference `$OPENCODE_FSM_RUNNER_ROOT/.opencode_fsm/generic_rollout.py` or `$OPENCODE_FSM_RUNNER_ROOT/.opencode_fsm/generic_evaluation.py` (wrong path).\n"
        "- It is OK to set up an isolated environment via `.opencode_fsm/bootstrap.yml` (e.g., venv + requirements) if required and deterministic.\n"
        "- If you use docker/compose, write all docker commands in `.opencode_fsm/stages/*.sh` and record how to stop them in `deploy_teardown.sh`.\n"
        "- Keep workloads bounded; use small smoke settings when possible.\n"
        "\n"
        "Use this minimal skeleton (edit as needed):\n"
        "```yaml\n"
        "version: 1\n"
        "\n"
        "security:\n"
        "  mode: safe\n"
        "  max_cmd_seconds: 7200\n"
        "  max_total_seconds: 86400\n"
        "\n"
        "tests:\n"
        "  cmds:\n"
        "    - bash .opencode_fsm/stages/tests.sh\n"
        "\n"
        "deploy:\n"
        "  setup_cmds:\n"
        "    - bash .opencode_fsm/stages/deploy_setup.sh\n"
        "  health_cmds:\n"
        "    - bash .opencode_fsm/stages/deploy_health.sh\n"
        "  teardown_policy: on_failure\n"
        "  teardown_cmds:\n"
        "    - bash .opencode_fsm/stages/deploy_teardown.sh\n"
        "\n"
        "rollout:\n"
        "  run_cmds:\n"
        "    - bash .opencode_fsm/stages/rollout.sh\n"
        "\n"
        "evaluation:\n"
        "  run_cmds:\n"
        "    - bash .opencode_fsm/stages/evaluation.sh\n"
        "  metrics_path: .opencode_fsm/metrics.json\n"
        "  required_keys: [score, ok]\n"
        "\n"
        "benchmark:\n"
        "  run_cmds:\n"
        "    - bash .opencode_fsm/stages/benchmark.sh\n"
        "\n"
        "artifacts:\n"
        "  out_dir: .opencode_fsm/artifacts\n"
        "```\n"
        "\n"
        "Tooling:\n"
        "- Use REAL OpenCode XML tool calls (do not wrap them in markdown fences).\n"
        "- READ format: `<read filePath=\"PATH\" />`\n"
        "- WRITE format: `<write filePath=\"PATH\">...file content...</write>`\n"
        "- BASH format (self-closing): `<bash command=\"...\" description=\"...\" />`\n"
        "- For bash, use the self-closing form exactly (`/>`). Do NOT use `<bash>...</bash>`.\n"
        "- Do NOT print pseudo tool snippets as plain text.\n"
        "- Do NOT fabricate `<tool_result>` blocks or simulated tool transcripts.\n"
        "Prefer file WRITE tool calls for any edits/creates (do NOT use bash redirections like `>` to write files).\n"
        "\n"
        "What to create/update:\n"
        f"- `{pipeline_rel}` (required; must be parseable YAML)\n"
        "- `.opencode_fsm/stages/tests.sh`\n"
        "- `.opencode_fsm/stages/deploy_setup.sh`\n"
        "- `.opencode_fsm/stages/deploy_health.sh`\n"
        "- `.opencode_fsm/stages/deploy_teardown.sh`\n"
        "- `.opencode_fsm/stages/rollout.sh`\n"
        "- `.opencode_fsm/stages/evaluation.sh`\n"
        "- `.opencode_fsm/stages/benchmark.sh`\n"
        "- Recommended: `.opencode_fsm/bootstrap.yml` for deterministic, non-interactive setup (especially for Python dependency installs).\n"
        "- Required output file: `.opencode_fsm/runtime_env.json` (written by deploy_setup)\n"
        "\n"
        "Two-tier behavior:\n"
        "A) Prefer REAL execution: if the repo clearly provides commands, wire them into the stage scripts.\n"
        "B) Otherwise FAIL-FAST: write the required JSON files with `ok=false` and a clear `reason`, then exit non-zero so the runner can trigger auto-repair.\n"
        "\n"
        "Hard constraints:\n"
        f"1) You may ONLY write `{pipeline_rel}` and files under `.opencode_fsm/`.\n"
        "2) Do NOT modify application code or any other repo files.\n"
        "3) Keep everything non-interactive (assume unattended strict mode).\n"
        "4) Do NOT hardcode repo-specific names/ports/URLs/paths unless the repo already defines them.\n"
        "5) Prefer putting complex logic into `.opencode_fsm/stages/*.sh`; keep `pipeline.yml` simple and stable.\n"
        "\n"
        "Discovery checklist (in this order):\n"
        "1) READ: README*, Makefile, pyproject.toml, requirements*.txt, package.json, docker-compose*.yml, .github/workflows/*\n"
        "2) Use `rg -n` to search for: test, pytest, npm test, make test, docker compose, benchmark, evaluate, rollout\n"
        "3) Only run lightweight commands when needed (e.g. `python -V`, `pytest --version`, `npm -v`, `make -n test`).\n"
        "4) Avoid long-running training; prefer bounded smoke commands if possible.\n"
        "\n"
        "Bootstrap guidance (generic):\n"
        "- If the repo appears to be Python (e.g., has `pyproject.toml` or `requirements*.txt`), create `.opencode_fsm/bootstrap.yml`.\n"
        "  It should create an isolated venv under `.opencode_fsm/venv`, install deps deterministically, and ensure later stages use the venv.\n"
        "  Use `uv` by default (faster, deterministic, and can install a compatible Python version without sudo).\n"
        "  Recommended pattern (pick a Python version compatible with the repo; if unclear, default to 3.11):\n"
        "    - `uv python install 3.11`\n"
        "    - `uv venv --python 3.11 .opencode_fsm/venv`\n"
        "    - If `requirements.txt` exists: `uv pip install --python .opencode_fsm/venv/bin/python -r requirements.txt`\n"
        "      Else: `uv pip install --python .opencode_fsm/venv/bin/python -e .`\n"
        "    - Set env: `PATH: .opencode_fsm/venv/bin:$PATH` and `OPENCODE_FSM_PYTHON: .opencode_fsm/venv/bin/python`\n"
        "  If `uv` is not available, fall back to `python3 -m venv` + `pip`, but do NOT use sudo.\n"
        "- If a `pip install ...` fails with `No matching distribution found`, do NOT keep guessing random package names/versions.\n"
        "  Either remove the unnecessary dependency, install without pinning, or pick a version that actually exists (pip often prints a list).\n"
        "- If Docker is required for any stage/hint, ensure scripts fail clearly when Docker is unavailable.\n"
        f"{hints_block}"
        "\n"
        "Acceptance requirements:\n"
        "- `pipeline.yml` must be `version: 1`.\n"
        "- `deploy_setup.sh` must write a JSON object to `.opencode_fsm/runtime_env.json`.\n"
        "- `rollout.sh` must write a JSON object to `.opencode_fsm/rollout.json`.\n"
        "- `rollout.json` MUST include `paths.samples_jsonl` pointing to a JSONL file with at least one valid sample line (keys: prompt, completion, reward).\n"
        "- At least one sample line MUST have a non-empty `completion` string (not just whitespace), or the rollout contract will fail.\n"
        "- `rollout.sh` and `evaluation.sh` must support `OPENCODE_RUNTIME_ENV_PATH` and local/remote inference inputs (`OPENCODE_TRAINED_MODEL_DIR` OR `OPENCODE_LLM_KIND=remote` + `OPENCODE_LLM_MODEL`, with endpoint/auth from env like `OPENAI_API_KEY`).\n"
        "- Evaluation must write a JSON object to `.opencode_fsm/metrics.json`.\n"
        "- `metrics.json` MUST include `ok` (boolean) and `score` (number). Exit 0 ONLY when `ok=true` and the score is from real evaluation.\n"
        "- Do NOT hardcode score values (e.g., `score=0.42`). Derive score from real benchmark execution and parse its outputs.\n"
        "- If evaluation cannot run, write `ok=false` + `reason`, then exit non-zero (do NOT emit placeholder success).\n"
        f"{required_line}"
        "\n"
        f"REPO_ROOT: {repo}\n"
        f"PIPELINE_PATH: {pipeline_rel}\n"
        "\n"
        "Now: inspect the repo and implement the contract.\n"
    )

def make_scaffold_contract_retry_prompt(
    repo: Path,
    *,
    pipeline_rel: str,
    require_metrics: bool,
    attempt: int,
    max_attempts: int,
    previous_failure: str,
    command_hints: list[str] | None = None,
) -> str:
    """Build a stricter retry prompt when the previous scaffold attempt produced no valid contract."""
    base = make_scaffold_contract_prompt(
        repo,
        pipeline_rel=pipeline_rel,
        require_metrics=require_metrics,
        command_hints=command_hints,
    )
    return (
        base
        + "\n"
        + "[RETRY]\n"
        + f"Attempt: {int(attempt)}/{int(max_attempts)}\n"
        + f"Previous failure: {tail(str(previous_failure or 'unknown'), 3000)}\n"
        + "If the previous failure mentions `invalid_redirect` or shows `2&&1`, you MUST fix it by replacing ALL `2&&1` with `2>&1` in the referenced stage scripts.\n"
        + "Your previous output did NOT produce a valid contract.\n"
        + "Do NOT stop at analysis.\n"
        + "Do NOT output pseudo tool syntax, and do NOT emit fake `<tool_result>` blocks.\n"
        + "Immediately execute REAL WRITE tool calls to create/update:\n"
        + f"- `{pipeline_rel}`\n"
        + "- `.opencode_fsm/stages/tests.sh`\n"
        + "- `.opencode_fsm/stages/deploy_setup.sh`\n"
        + "- `.opencode_fsm/stages/deploy_health.sh`\n"
        + "- `.opencode_fsm/stages/deploy_teardown.sh`\n"
        + "- `.opencode_fsm/stages/rollout.sh`\n"
        + "- `.opencode_fsm/stages/evaluation.sh`\n"
        + "- `.opencode_fsm/stages/benchmark.sh`\n"
        + "This retry is successful only if `pipeline.yml` exists and passes contract validation.\n"
    )

def make_fix_or_replan_prompt(
    step: dict[str, str],
    verify: VerificationResult,
    *,
    tests_cmds: list[str],
    artifacts_dir: Path,
) -> str:
    metrics_errors = verify.metrics_errors or []
    metrics_block = ""
    if verify.metrics_path or metrics_errors:
        metrics_block = (
            "[METRICS]\n"
            f"metrics_path: {verify.metrics_path}\n"
            f"errors: {metrics_errors}\n"
            f"metrics_preview: {tail(json.dumps(verify.metrics or {}, ensure_ascii=False), 2000)}\n"
            "\n"
        )
    return (
        "Verification failed. You must choose exactly ONE:\n"
        "A) Fix code/scripts/manifests until verification passes (preferred).\n"
        "B) If it truly cannot be closed without missing info: ONLY edit PLAN.md to split the step or mark it Blocked.\n"
        "\n"
        f"FAILED_STEP: (STEP_ID={step['id']}) {step['text']}\n"
        "\n"
        f"FAILED_STAGE: {verify.failed_stage}\n"
        f"TEST_CMDS: {' && '.join(tests_cmds)}\n"
        f"ARTIFACTS_DIR: {artifacts_dir}\n"
        "\n"
        "You MAY create new files if needed; this run is unattended and file-create/add-to-chat prompts are auto-approved.\n"
        "\n"
        "If this is an environment/tooling/auth issue, you may write `.opencode_fsm/actions.yml` for the runner to execute.\n"
        "actions.yml format (YAML):\n"
        "version: 1\n"
        "actions:\n"
        "- id: fix-001\n"
        "  kind: run_cmd\n"
        "  cmd: <shell command>\n"
        "  timeout_seconds: 300\n"
        "  retries: 0\n"
        "  risk_level: low|medium|high\n"
        "  rationale: <why>\n"
        "Notes:\n"
        "- In strict unattended mode, avoid interactive login commands (e.g. `docker login` without non-interactive flags).\n"
        "- Runner records artifacts and then deletes actions.yml.\n"
        "\n"
        f"{fmt_stage_tail('BOOTSTRAP', verify.bootstrap)}"
        f"{fmt_stage_tail('AUTH', verify.auth)}"
        f"{fmt_stage_tail('TESTS', verify.tests)}"
        f"{fmt_stage_tail('DEPLOY_SETUP', verify.deploy_setup)}"
        f"{fmt_stage_tail('DEPLOY_HEALTH', verify.deploy_health)}"
        f"{fmt_stage_tail('ROLLOUT', verify.rollout)}"
        f"{fmt_stage_tail('EVALUATION', verify.evaluation)}"
        f"{fmt_stage_tail('BENCHMARK', verify.benchmark)}"
        f"{metrics_block}"
    )

def make_mark_done_prompt(step: dict[str, str]) -> str:
    return (
        "This step passed verification. ONLY edit PLAN.md:\n"
        f"1) Move `- [ ] (STEP_ID={step['id']}) ...` from Next to Done, and change it to `- [x]`.\n"
        "2) Pick ONE smallest atomic unchecked item from Backlog into Next (keep Next to exactly one item).\n"
        "3) If Backlog is empty, leave Next empty (keep the heading, no items).\n"
    )

def make_block_step_prompt(step: dict[str, str], last_failure: str) -> str:
    return (
        "Fix attempts exceeded the limit. ONLY edit PLAN.md:\n"
        "1) Remove the step from Next; in Notes, explain why it's Blocked and what human input is needed.\n"
        "2) Pick one item from Backlog into Next (or leave Next empty).\n"
        "\n"
        f"BLOCKED_STEP: (STEP_ID={step['id']}) {step['text']}\n"
        "\n"
        "[LAST_FAILURE]\n"
        f"{tail(last_failure, STDIO_TAIL_CHARS)}\n"
    )
