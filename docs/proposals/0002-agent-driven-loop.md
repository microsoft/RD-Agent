# ADR 0002 — The agent *is* the loop: RD-Agent machinery as deterministic tools

- **Status:** Accepted
- **Date:** 2026-07-06
- **Deciders:** repo owner
- **Relationship to 0001:** **Amends [ADR 0001](0001-autonomous-quant-rd-loop.md).** Keeps 0001's
  vision (autonomous qlib alpha search with honest evaluation), its guardrail design (§7), and its
  file map. **Supersedes 0001's execution-mode decision (§6):** we do *not* run RD-Agent's autonomous
  program and swap three LLM components. Instead, **this Claude Code session's agent drives the loop
  directly**, calling RD-Agent's deterministic machinery as tools.

> Like 0001, this is both a decision record and a cold-start handoff. A fresh session should be able
> to read this and continue. Paths are relative to the repo root.

---

## 1. TL;DR decision

RD-Agent's loop routes *all* intelligence through one choke point —
`APIBackend().build_messages_and_create_chat_completion(...)` — i.e. it phones a *second, external*
LLM for every propose / code / judge step. That is what needs `.env` + `ANTHROPIC_API_KEY`, and it is
why `python rdagent/app/qlib_rd_loop/quant.py` fails with no credentials.

**We already have an LLM with tools: this session's agent.** So instead of paying an API to run a
worse-integrated copy of the agent, **the agent becomes the brain** (propose hypotheses, write
factor/model code, judge results) and RD-Agent's **LLM-free machinery** — Dockerized qlib execution,
the trace ledger, and a statistical guardrail — is exposed as **deterministic tools the agent calls**.

Consequences: **no second API key**, every reasoning step is visible in the agent transcript, and two
of 0001's open problems dissolve (see §6). The cost: the loop is **attended and low-throughput**
(a handful of honest iterations per session) rather than **unattended at scale** (hundreds of loops
overnight). We chose transparency + honest evaluation over unattended throughput. If unattended scale
becomes the goal, 0001's program-driven path is the fallback (point LiteLLM at Claude via `.env`).

---

## 2. For the next session — start here

**What is true right now (verified this session):**
- **Slice 1 is proven.** A cold session drove a Dockerized qlib backtest end-to-end and got real
  metrics back, with **no `.env` and no LLM call in the path**. See §4.
- The tools exist: **`agent_loop/run_qlib.py`** (execute) and **`agent_loop/guardrail.py`** (the
  deterministic promote/reject gate). Both are LLM-free.
- **Phase B is done too:** the guardrail rejects a deliberately overfit factor and applies a real
  Deflated-Sharpe multiple-testing haircut. Tests: `agent_loop/tests/test_guardrail.py` (4 passing).
- `agent_loop/` is a **new top-level package kept outside `rdagent/`** so upstream merges stay cheap
  (0001 §10). Do not put loop-driving logic inside `rdagent/`.

**Environment facts (this machine, verified):**
- Host env: conda env **`rdagent`** (Python 3.10.20), `rdagent` editable-installed
  (`pip install -e .`). Run tools with `~/anaconda3/envs/rdagent/bin/python`. (The `base` env is
  Python 3.11 and is missing deps — do not use it.)
- qlib runs in Docker image **`local_qlib:latest`** (already built). Docker daemon is up.
- qlib data: **`~/.qlib/qlib_data/cn_data`** is extracted (features/instruments/calendars present).
  The container mounts `~/.qlib/` → `/root/.qlib/`.
- **Two env fixes are required and are baked into `run_qlib.py`:**
  1. Force **`env_type=docker`**. The default is `conda` (`MODEL_CoSTEER_ENV_TYPE`), which would
     auto-create a heavy `rdagent4qlib` conda env (qlib+torch). The Docker image is already provisioned.
  2. Pass **`MLFLOW_ALLOW_FILE_STORE=true`** into the container — its newer mlflow refuses the
     file-store tracking backend otherwise (hard-fails the run with no result file).
- GPU: this is a Mac; `QlibDockerConf.enable_gpu` defaults `True` but auto-disables when unavailable.
  `run_qlib.py` sets `QLIB_DOCKER_ENABLE_GPU=False` to skip the probe.

**Next deliverables (in order):**
1. ~~`guardrail` tool~~ — **DONE** (`agent_loop/guardrail.py`). See §5b.
2. **`trace`** tool — a simple JSON/SQLite DAG ledger the agent owns (exp + agent's qualitative
   feedback + guardrail stats). Schema is already defined in `guardrail.py` (the `trials` list it
   reads). Do *not* reuse `LoopBase.load`/checkpoint machinery — it is coupled to the program loop.
3. **factor injection path** — exercise `run_qlib.py --factors` with a real hand-written factor
   (`combined_factors_df.parquet`) so the agent can test its own alpha, not just the Alpha20 baseline.
   This is what produces the candidate + locked-holdout runs the guardrail consumes.
4. **Claude Code skill `quant-rd-loop`** — the one-iteration recipe wiring the agent + the three tools.

---

## 3. Architecture: who does what

| R&D stage | Brain | Hands (deterministic tool) |
|---|---|---|
| Propose / Formalize | **agent** (reads scenario + knowledge files + trace SOTA, writes hypothesis) | — |
| Implement | **agent** (writes factor expr / `model.py`) | — |
| Execute | — | **`agent_loop/run_qlib.py`** → `QlibFBWorkspace.execute()` → Docker qlib |
| Evaluate (decision) | — | **`guardrail`** (holdout + Deflated Sharpe + cost margin) — *not an LLM* |
| Evaluate (qualitative) | **agent** (observations, next-hypothesis) | — |
| Memorize | — | **`trace`** (JSON/SQLite DAG ledger) |

The agent is `HypothesisGen` + coder + the *qualitative* half of feedback. The promote/reject decision
is deterministic Python — which is exactly what 0001 Gap 1 demanded (stop letting an LLM eyeball
test-segment metrics). Here that is **structural**, not a swapped module.

---

## 4. Verified spine (slice 1)

**The whole execution path is LLM-free.** The only file under `rdagent/scenarios/qlib/developer/`
that imports `APIBackend` is `feedback.py` — the component 0001 replaces. Both runners
(`factor_runner.py`, `model_runner.py`) and `utils/env.py` are LLM-free. Every backtest reduces to:

```python
# rdagent/scenarios/qlib/experiment/workspace.py — QlibFBWorkspace.execute()
result, stdout = exp.experiment_workspace.execute(qlib_config_name="conf_*.yaml", run_env=env_to_use)
exp.result = result   # pandas Series, indexed by metric name
```

`run_env` carries `train_start/end, valid_start/end, test_start/end` and is passed as **container env
vars**, which qlib's `qrun` substitutes into the YAML template's Jinja fields. **This is the holdout
seam:** the guardrail controls which date segments a backtest is allowed to see, with no core edits.

**Proof run** (`conf_baseline.yaml`, Alpha20, train 2015–16 / valid 2017 / test 2018), metrics returned:

```
IC 0.0277   Rank IC 0.0283   ICIR 0.254   Rank ICIR 0.250
1day.excess_return_with_cost.annualized_return -0.0195
1day.excess_return_with_cost.max_drawdown      -0.0797
1day.excess_return_with_cost.information_ratio -0.245
```

(Weak numbers — it's a deliberately tiny baseline; the point is the spine.) The Series contains
**exactly** the `IMPORTANT_METRICS` from `feedback.py` (`IC`, `...with_cost.annualized_return`,
`...with_cost.max_drawdown`) plus Rank IC/ICIR/information_ratio — the guardrail's inputs.

---

## 5. The `run_qlib` tool (built)

`agent_loop/run_qlib.py` — `python -m agent_loop.run_qlib [flags]`. Key flags: `--conf` (template
yaml), `--template factor|model`, `--factors <parquet>` (inject `combined_factors_df.parquet`),
`--features <json>` (default Alpha20), the six date-segment flags, `--env docker|conda`, `--gpu`,
`--timeout`, `--out <json>`. Returns/dumps `{workspace, conf, segments, metrics}`. It mutates
`MODEL_COSTEER_SETTINGS.env_type` at runtime and injects `MLFLOW_ALLOW_FILE_STORE`/GPU env — a caller
never needs a `.env`.

---

## 5b. The `guardrail` tool (built)

`agent_loop/guardrail.py` — `python -m agent_loop.guardrail --candidate <sel.json> --holdout <hold.json>
[--trace <trace.json>] [--sota <sel.json>]`. Deterministic promote/reject; **no LLM**. Promotes only if
**all** gates pass:

1. **holdout_ok** — beats the current SOTA on a locked holdout segment (or clears a floor if no incumbent).
2. **dsr_ok** — Deflated Sharpe Ratio (Bailey & López de Prado 2014) ≥ threshold (default 0.95). DSR
   reads the raw daily excess-return-with-cost series from `<workspace>/ret.pkl` for honest T / skew /
   kurtosis, then haircuts the Sharpe by the trial count `N = len(trace.trials)+1` and the trial-Sharpe
   dispersion. As `N` grows, the bar rises.
3. **net_positive** — net-of-cost information ratio > 0 (reject pure gross winners).
4. **beats_sota_net** — net-of-cost IR beats the incumbent by `--cost_margin`.

**Acceptance results** (`agent_loop/tests/test_guardrail.py`, 4 passing): an overfit factor (strong
selection Sharpe, poor holdout Rank IC) is rejected on the holdout gate; and a Sharpe-2.2 candidate that
is promoted at `N=1` (DSR 0.98) is rejected after 200 trials (DSR 0.35) — the multiple-testing haircut
biting exactly as designed. The trace JSON schema the guardrail reads is the contract for the Phase C
`trace` tool.

## 6. What dissolves vs 0001

- **Embeddings / RAG (0001 §8, §10 open item): gone.** RAG existed to feed a *remote* LLM past
  context. The agent retrieves by reading the knowledge-substrate files directly. Anthropic-has-no-
  embeddings is a non-problem here.
- **LLM-judge eval gap (0001 §7, Gap 1): structural, not a swap.** The decision is deterministic Python
  the agent calls; there is no LLM in the promote/reject path by construction.
- **Still true from 0001:** Docker dependency stays (qlib runs in a container — but that is local
  execution, not an API call). The guardrail's statistical design (holdout + Deflated Sharpe + cost
  margin, purged CV later) is unchanged — see 0001 §7.

---

## 7. What we give up (and the fallback)

Program-driven RD-Agent buys **unattended overnight autonomy**: parallel loops, bandit factor-vs-model
selection, auto-retry evolving coder, hundreds of iterations. Agent-driven is **attended and
low-throughput** by nature. If throughput/scale becomes the goal, the fallback is 0001's path: keep
`quant.py`, point `LiteLLMAPIBackend` at Claude via `.env`, and swap the summarizer for the guardrail
module. The two paths **share** the guardrail and the knowledge substrate, so that work is not wasted.

---

## 8. Phased plan (supersedes 0001 §9 Phases 0–2)

- **Phase A — spine (DONE).** `run_qlib` tool; slice-1 baseline backtest returns real metrics, no key.
- **Phase B — guardrail tool (DONE).** Deterministic holdout + Deflated Sharpe + cost margin (0001 §7
  v0), consuming `run_qlib` metric JSON. *Gate met: overfit factor rejected; DSR haircut blocks a
  marginal winner as trial count grows.* See §5b.
- **Phase C — trace tool.** JSON/SQLite DAG ledger owned by `agent_loop`. *Gate: sessions resume.*
- **Phase D — factor path.** Hand-written factor via `--factors`; agent tests its own alpha end-to-end.
- **Phase E — skill.** `quant-rd-loop` Claude Code skill = the one-iteration recipe over A–D.
- **Phase F — knowledge substrate** (0001 §Gap 2 / Phase 3): curated qlib operator/handler/model-card
  corpus the agent reads directly. Later: purged/embargoed CV (0001 §7 v1).

---

## 9. Operational model — long-running jobs without stalling the session

**The core tension of agent-driven execution:** every loop iteration pairs *cheap* agent reasoning
(propose / code / judge — seconds) with an *expensive* deterministic job (a Dockerized qlib backtest is
2–4 min today on a short segment; a full universe, GPU model training, or a purged-CV sweep will be tens
of minutes to hours). If the session blocks on each job, the agent — the scarce resource — sits idle.
This is a first-class design constraint, not an afterthought, and it shapes how the loop is run.

**How the loop stays active while a job runs:**

- **Background + notify, never block.** Launch backtests as background jobs (`run_in_background`); the
  harness re-invokes the agent when the job exits. The agent does *useful* filler work in the gap:
  propose the next hypothesis, write/refine a tool, update this ADR, or do git/PR housekeeping. (This
  very document was extended, and the branch/PR prepared, while a 4-backtest Phase-D run was executing.)
- **Parallel fan-out.** Independent backtests can run as concurrent background jobs (e.g. baseline vs
  candidate × selection vs holdout — 4 at once). Caveat: qlib's Docker env mounts a shared cache
  (`/tmp/full` → `workspace_cache`); heavy parallelism can race on it, so cap concurrency or isolate
  caches per job.
- **File-based state = interruptible/resumable.** The loop's state is *on disk* (run JSONs, the trace
  ledger), not in a live Python process. A session can end mid-flight and a fresh one resumes from the
  ledger — a structural advantage over the program-driven loop, whose state lives in `LoopBase` memory
  and needs its checkpoint machinery. Design tools to be **idempotent and re-entrant**: write results to
  a stable path, and treat "job already produced its JSON" as success.
- **Polling only when the harness can't notify.** For state the harness cannot observe (a CI run, a
  remote queue), poll with a `Monitor` until-loop. Do **not** hand-roll `sleep` in a foreground shell —
  the harness blocks long foreground sleeps by design.
- **Self-paced loops.** For an unattended cadence, schedule the next iteration with a wake-up timer.
  Match the delay to what you're waiting on and to the prompt-cache window (~5 min): sub-5-min polls
  keep the cache warm for a job about to finish; for long jobs, sleep long (20–60 min) and let the job's
  own completion event be the primary wake signal, with the timer as a fallback.

**Practical gotchas discovered during bring-up (all now handled in `agent_loop/`):**

- Foreground `sleep` is blocked — use background jobs or a `Monitor` until-loop to wait on a condition.
- On macOS, qlib/`D.features` uses the `spawn` multiprocessing start method; any host-side factor-
  computation script **must** guard its body with `if __name__ == "__main__":` or it crashes with
  `freeze_support()` / bootstrapping errors.
- A background command whose stdout is filtered through a pipe can look "empty" mid-run; write full
  output to a log file and inspect that, rather than trusting a grep'd tail.

**Implication for throughput.** Even with backgrounding, agent-driven is bounded by *attended* wall
time: the agent must be re-invoked to advance. That is the deliberate trade in §7. The mitigations above
(parallel fan-out + filler work + resumable file state) recover a lot of it, but if the goal becomes
"hundreds of iterations unattended overnight," the program-driven fallback (§7) is the right tool.

---

## Appendix — reproduce slice 1

```bash
PY=~/anaconda3/envs/rdagent/bin/python
$PY -m agent_loop.run_qlib --conf conf_baseline.yaml \
  --train_start 2015-01-01 --train_end 2016-12-31 \
  --valid_start 2017-01-01 --valid_end 2017-12-31 \
  --test_start 2018-01-01 --test_end 2018-12-31 --out /tmp/res.json
# Needs: Docker up, local_qlib:latest image, ~/.qlib/qlib_data/cn_data extracted.
# Does NOT need: .env, ANTHROPIC_API_KEY, or any LLM endpoint.
```
