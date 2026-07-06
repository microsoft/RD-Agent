# ADR 0001 — Fork RD-Agent as the engine for an Autonomous Quant R&D Loop

- **Status:** Proposed
- **Date:** 2026-07-06
- **Deciders:** repo owner (fork of `microsoft/RD-Agent`)
- **Supersedes:** n/a
- **Context source:** analysis of `microsoft/RD-Agent` @ HEAD (commit dated 2026-05-06, MIT license) and `microsoft/qlib`.

> This document is both a **decision record** and a **cold-start handoff brief**. A fresh
> Claude Code session opened in the fork should be able to read this top-to-bottom and
> continue the work without re-deriving anything. All file paths below are relative to the
> RD-Agent repo root unless noted.

---

## 1. TL;DR decision

**Fork/extend `microsoft/RD-Agent`; do NOT rebuild the R&D loop from scratch.**

RD-Agent already implements the entire loop we want (propose → code → run qlib → evaluate →
record), Dockerized qlib execution, a DAG research ledger, HITL hooks, checkpoint/resume, and
factor+model joint optimization — ~65K LOC, MIT-licensed, actively maintained. Every component
is swappable via a dotted-class-path config (`import_class(PROP_SETTING.*)`).

Our differentiation goes into **three pluggable slots**, not a rewrite:

1. **Statistical eval gate** — replace the LLM-judge `summarizer` with an overfitting-aware
   feedback module (locked out-of-sample holdout + trial-count-aware Deflated Sharpe). **This is
   the highest-ROI change and the first deliverable.**
2. **Curated qlib knowledge substrate** — feed proposals a retrievable "capability wiki"
   (operators, handlers, model cards, metric semantics), not just prompt-template scenario text.
3. **Claude backend + optional skill/MCP surface** — point the LiteLLM backend at Claude.

Build from scratch **only** if the research goal is the loop/methodology itself. If
qlib-driven autonomous alpha research is the goal and the loop is a means: **fork.**

---

## 2. For the next session — start here

**Goal of the project:** an autonomous quant R&D loop that treats qlib as an experiment-execution
engine and iteratively searches for alpha (factors + models), with *scientifically honest*
evaluation.

**First milestone (do this before anything else):**
1. Reproduce the RD-Agent quant baseline end-to-end on this machine so we have a known-good loop.
2. Point the LLM backend at Claude (LiteLLM).
3. Land **one** change: swap the factor/model `summarizer` for a guardrail-aware feedback module
   (§7). This converts a p-hacking-prone loop into something defensible and validates the fork
   approach in days.

**Environment quick facts (verified):**
- Python 3.10, `conda create -n rdagent python=3.10` then `pip install -e .` in the fork.
- **qlib runs inside Docker** (`rdagent/utils/env.py` — `DockerEnv`; there is also a `LocalEnv`).
  Docker must be available. First run pulls an image.
- LLM backend is **LiteLLM** (multi-provider). Config via `.env` (see §8).
- Run the joint factor+model loop: `rdagent fin_quant`  (factor-only: `rdagent fin_factor`,
  model-only: `rdagent fin_model`, factor-from-report: `rdagent fin_factor_report`).
- Direct/dev invocation: `python rdagent/app/qlib_rd_loop/quant.py`.
- Resume a session: `rdagent fin_quant --path <LOG_PATH>/__session__/<i>/<step> --step_n 1`.
- Health check: `rdagent health_check`.

**Read these files first (in order):**
1. `rdagent/components/workflow/rd_loop.py` — the generic loop (`RDLoop`).
2. `rdagent/app/qlib_rd_loop/quant.py` — the quant (factor+model) loop (`QuantRDLoop`).
3. `rdagent/app/qlib_rd_loop/conf.py` — the pluggable config (`QuantBasePropSetting`).
4. `rdagent/scenarios/qlib/developer/feedback.py` — the current evaluation (what we replace).
5. `rdagent/core/proposal.py` — the interfaces (`Experiment2Feedback`, `Trace`, `HypothesisFeedback`).

---

## 3. Background: the vision, formalized

The unit of work is a **Hypothesis** that formalizes to a qlib experiment:

```
h = (rationale, factor_exprs, model_spec, dataset_spec, eval_plan)
```

The **loop** (and how it already maps onto RD-Agent's `QuantRDLoop`):

| R&D stage        | Our name           | RD-Agent method (`quant.py`) | What happens                                             |
|------------------|--------------------|------------------------------|---------------------------------------------------------|
| Ideate+Formalize | propose            | `direct_exp_gen`             | `hypothesis_gen.gen()` → `hypothesis2experiment.convert()` |
| Implement        | code               | `coding`                     | LLM writes factor/model code (`factor_coder`/`model_coder`) |
| Execute          | run                | `running`                    | qlib runs in Docker; code injected into YAML templates   |
| Evaluate+Reflect | feedback           | `feedback`                   | `summarizer.generate_feedback()` → `HypothesisFeedback`  |
| Memorize         | record             | `record`                     | `trace.sync_dag_parent_and_hist()` — DAG ledger          |

The agent optimizes an objective (OOS Rank ICIR / portfolio IR) over the search space
(factors × models × datasets) **subject to guardrails**. "Brainstorming alpha ideas itself" is
just the Ideate stage running unconditioned instead of conditioned on a user direction.

---

## 4. Verified findings about RD-Agent

**Layering (LOC):** `core/` ~2.0K (abstractions), `components/` ~12.5K (reusable building blocks:
coder, runner, proposal, workflow, knowledge_management, loader, benchmark), `scenarios/` ~30K
(qlib, data_science, kaggle, rl, general_model, finetune), `app/` ~6.9K (entry points),
`oai/` ~1.9K (LLM backend), `utils/` ~3.9K (incl. Docker env + qlib helpers). Total ~65K.

**Fully pluggable.** `QuantBasePropSetting` (`app/qlib_rd_loop/conf.py`, env prefix `QLIB_QUANT_`)
declares every component as a dotted class path resolved by `import_class(...)`. Overridable by
env var. Slots: `quant_hypothesis_gen`, `factor_hypothesis2experiment`, `model_hypothesis2experiment`,
`factor_coder`, `model_coder`, `factor_runner`, `model_runner`, `factor_summarizer`,
`model_summarizer`, plus `action_selection ∈ {bandit, llm, random}` and the train/valid/test dates.

**Already built (do not rebuild):** Dockerized qlib execution with retries/volume-mounts
(`utils/env.py`); DAG research ledger (`Trace` in `core/proposal.py`, `trace.hist` is a list of
`(Experiment, feedback)`); HITL via multiprocessing queues (`_interact_hypo`, `_interact_feedback`);
checkpoint/resume (`RDLoop.load`, `--step_n/--loop_n/--all_duration`); parallel loops
(`get_max_parallel`); factor+model **joint** optimization with bandit action selection; a
knowledge base with vector store + knowledge graph (`components/knowledge_management/vector_base.py`,
`graph.py`) used for RAG over past coding attempts; LiteLLM multi-provider backend (`rdagent/oai/`).

**Execution mechanics.** Factor/model code the LLM writes is dropped into qlib workflow YAML
templates (`scenarios/qlib/experiment/factor_template/conf_*.yaml`,
`.../model_template/`) and run via qlib inside the container. `exp.result` is a pandas object
indexed by metric name.

---

## 5. The two gaps = our differentiation

### Gap 1 (critical): no statistical overfitting guardrail
`scenarios/qlib/developer/feedback.py` decides "is this better than the best-so-far" by having an
**LLM compare metrics**:

```python
IMPORTANT_METRICS = ["IC",
                     "1day.excess_return_with_cost.annualized_return",
                     "1day.excess_return_with_cost.max_drawdown"]
...
decision = convert2bool(response_json.get("Replace Best Result", "no"))  # LLM judgment
```

Across hundreds of trials, "SOTA" is selected by an LLM eyeballing **test/backtest-segment**
metrics. There is **no** Deflated Sharpe, **no** multiple-testing correction, and **no** locked
out-of-sample holdout the loop is forbidden to select on. This is textbook selection-on-the-test-set
/ p-hacking exposure. It is the single biggest scientific weakness — and it lives entirely in a
pluggable component.

### Gap 2: the "wiki" is prompt-templates, not a curated capability substrate
Proposal context comes from `scenarios/qlib/prompts.yaml` scenario descriptions plus RAG over past
*experience*. There is no curated, retrievable corpus of qlib's expression operators, handler
catalog (Alpha158/Alpha360), model cards, or metric semantics — the substrate the vision calls for.

### (Not a gap, a preference) Not Claude-Code-native
It's a standalone framework (`LoopBase`, `pydantic-settings`, `fire`/`typer`, hard Docker dep,
prompt-YAML system). Forking means adopting these conventions. Acceptable given the head start.

---

## 6. Decision & alternatives considered

**Decision:** Fork. Vendor RD-Agent's loop + qlib scenario as the engine; add differentiation via
the three plug-points.

**Alternatives considered:**
- **Build fresh in a Claude-Code-native stack (skills + MCP + custom loop).** Rejected: rebuilds
  the hard, non-differentiating 80% (Docker qlib exec, coder, runner, trace, HITL, resume) —
  months of work, and we'd rediscover the same failure modes (uncompilable factors, env setup).
  Choose this only if the *loop itself* is the research contribution.
- **Contribute upstream to RD-Agent instead of forking.** Deferred: guardrails and a curated
  substrate are opinionated research changes; iterate in a fork first, upstream later if desired.
- **Use RD-Agent unmodified.** Rejected: the guardrail gap makes results non-defensible for our
  purposes.

**Consequences:** we inherit RD-Agent's conventions and Docker dependency and track upstream via
periodic merges; in exchange we get a working loop on day one and confine our work to three modules.

---

## 7. First deliverable — the guardrail feedback module

Replace the LLM-judge summarizer with an **overfitting-aware `Experiment2Feedback`**. Interface
(verified in `core/proposal.py`):

```python
class Experiment2Feedback(ABC):
    def __init__(self, scen: Scenario) -> None: ...
    @abstractmethod
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback: ...

# HypothesisFeedback fields: observations, hypothesis_evaluation, new_hypothesis, reason, decision(bool)
# decision == "replace the current SOTA with this experiment"
# trace.hist : list[(Experiment, feedback)]   → number of trials so far = len(trace.hist)
# trace.get_sota_hypothesis_and_experiment()  → current best
# exp.result : pandas object indexed by metric name (see IMPORTANT_METRICS)
```

**Design (phased):**

- **v0 — trial-count-aware promotion gate (cheapest, do first).** Wrap the existing LLM feedback
  (keep its qualitative `observations`/`new_hypothesis` — they guide the next proposal well), but
  **override `decision`** with a statistical rule:
  - Maintain a **locked holdout** the loop never selects on. Concretely: carve a final segment
    after `test_end` (e.g. reserve 2019–2020 as OOS, train/select on ≤2018), or extend the data and
    hold out the last N months. The LLM sees only the selection segment; the gate scores the holdout.
  - Compute a **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) using `N = len(trace.hist)`
    (number of trials) and the variance of trial Sharpes accumulated in the trace. Promote only if
    `DSR > threshold` **and** the holdout metric beats SOTA **and** the improvement survives a
    cost-adjusted margin (turnover-aware, using the `_with_cost` metrics).
  - `decision = (holdout_beats_sota AND dsr_ok AND cost_margin_ok)`.
- **v1 — purged/embargoed cross-validation.** Replace the single train/valid/test split with
  purged K-fold + embargo to kill leakage across the label horizon (label is
  `Ref($close,-2)/Ref($close,-1)-1`, so embargo ≥ 2 days). This requires touching how segments are
  materialized in the qlib templates (`scenarios/qlib/experiment/*/conf_*.yaml`) and the runner.
- **v2 — record the guardrail stats in the trace** so proposals and the dashboard can see
  trial count, DSR, and holdout-vs-selection gaps (detects overfitting drift over the run).

**Wiring (no core edits needed):** set env vars to point the summarizer slots at the new class:
```bash
QLIB_QUANT_FACTOR_SUMMARIZER=<fork_pkg>.guardrail.GuardrailFactorFeedback
QLIB_QUANT_MODEL_SUMMARIZER=<fork_pkg>.guardrail.GuardrailModelFeedback
```
(or edit the defaults in `app/qlib_rd_loop/conf.py`). Subclass the existing
`QlibFactorExperiment2Feedback`/`QlibModelExperiment2Feedback` to reuse their prompt plumbing.

**Acceptance test:** run ~30–50 loop iterations; confirm (a) promotions require holdout
improvement, (b) DSR drops as trial count grows and blocks marginal "winners," (c) a deliberately
overfit factor (e.g. a high in-sample IC, noisy OOS) is *rejected*.

---

## 8. Claude backend wiring

RD-Agent defaults to LiteLLM. Point it at Claude via `.env` at repo root. The README shows the
OpenAI-style shape (`CHAT_MODEL`, `EMBEDDING_MODEL`, `OPENAI_API_BASE`). For Claude through
LiteLLM, use LiteLLM's Anthropic provider naming and `ANTHROPIC_API_KEY`, e.g.:

```bash
# .env  (verify exact keys against rdagent/oai/ and the RD-Agent LiteLLM docs before relying on it)
CHAT_MODEL=claude-opus-4-8            # or the LiteLLM-qualified name, e.g. anthropic/claude-...
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_MODEL=text-embedding-3-small  # embeddings may need a separate provider; the knowledge
                                        # base uses embeddings — decide provider or disable RAG
```

**Open item for the next session:** confirm the exact LiteLLM backend selector and env keys in
`rdagent/oai/` (there is a backend setting) and how embeddings are configured, since the
knowledge base needs an embedding model. Anthropic has no embeddings API — either route embeddings
to another provider or make the KB embedding-optional.

---

## 9. Phased plan

- **Phase 0 — Reproduce baseline.** Install fork, Docker up, qlib data ready, `rdagent fin_quant`
  runs a few loops on the stock config. Confirm you can read `exp.result` and the trace log. *Gate:
  a clean baseline run.*
- **Phase 1 — Claude backend.** LiteLLM → Claude; resolve embeddings. *Gate: loop runs on Claude.*
- **Phase 2 — Guardrail v0 (first real deliverable).** Locked holdout + trial-aware DSR gate as a
  drop-in summarizer. *Gate: overfit factor rejected; promotions require OOS improvement.*
- **Phase 3 — Knowledge substrate.** Curated qlib capability corpus (operators, Alpha158/360
  handlers, model cards, metric glossary) injected into the proposal context / KB. *Gate: proposals
  reference real operators/fields and stop hallucinating expressions.*
- **Phase 4 — Guardrail v1 (purged CV).** Leakage-safe evaluation in the qlib templates/runner.
- **Phase 5 — Interface/dashboard.** Read the trace + guardrail stats; hypothesis tree, OOS-vs-
  selection gaps, DSR over trials. (RD-Agent ships a `rdagent ui` / `server_ui` — extend rather
  than rebuild.)
- **Phase 6 — Optional Claude-native surface.** Wrap the loop / qlib tools as an MCP server or
  Claude Code skills if a Claude-native operator UX is wanted.

---

## 10. Risks & open questions

- **Upstream drift.** RD-Agent is active; keep the fork's changes confined to the plug-in modules
  and a thin `conf.py` override to make periodic merges cheap. Avoid editing `core/`.
- **Docker dependency** is hard (qlib runs in a container). CI and any cloud runs need Docker.
- **Embeddings/KB provider** unresolved with a Claude chat backend (Anthropic has no embeddings) —
  decide provider or make RAG optional (see §8).
- **DSR inputs.** Deflated Sharpe needs the trial count and the distribution/variance of trial
  Sharpes; ensure the trace accumulates enough per-trial stats. May need to persist Sharpe (not
  just IC/return/drawdown) per experiment.
- **Cost.** Each loop iteration = LLM calls + a full Dockerized qlib backtest. Budget and cap
  `loop_n`; use cheap pre-checks before full backtests where possible.
- **"test" segment reuse.** The stock config selects on the 2017–2020 test segment. The guardrail
  redefines what the loop is allowed to select on; make sure the holdout is genuinely untouched by
  proposal context and feedback.

---

## Appendix A — Key file map (RD-Agent)

| Path | Role |
|---|---|
| `rdagent/components/workflow/rd_loop.py` | Generic loop `RDLoop` (steps: direct_exp_gen, coding, running, feedback, record) |
| `rdagent/app/qlib_rd_loop/quant.py` | `QuantRDLoop` — factor+model joint loop + `main()` |
| `rdagent/app/qlib_rd_loop/{factor,model,factor_from_report}.py` | Single-mode loops |
| `rdagent/app/qlib_rd_loop/conf.py` | `QuantBasePropSetting` etc. — the pluggable config (env prefix `QLIB_QUANT_`) |
| `rdagent/app/cli.py` | `rdagent fin_quant / fin_factor / fin_model / fin_factor_report / ui` |
| `rdagent/core/proposal.py` | `Hypothesis`, `HypothesisFeedback`, `Trace`, `Experiment2Feedback`, `HypothesisGen`, `Hypothesis2Experiment` |
| `rdagent/scenarios/qlib/developer/feedback.py` | Current LLM-judge evaluation (**replace this**) |
| `rdagent/scenarios/qlib/developer/{factor_runner,model_runner}.py` | Run qlib for factor/model |
| `rdagent/scenarios/qlib/proposal/{quant,factor,model}_proposal.py` | Hypothesis generators |
| `rdagent/scenarios/qlib/experiment/*/conf_*.yaml` | qlib workflow templates (where code is injected) |
| `rdagent/scenarios/qlib/prompts.yaml` | Scenario/context prompts (Gap 2 substrate hook) |
| `rdagent/components/knowledge_management/{vector_base,graph}.py` | Vector store + knowledge graph (RAG) |
| `rdagent/utils/env.py` | Docker/Local execution env for qlib |
| `rdagent/utils/qlib.py` | qlib helpers (`ALPHA20`, `validate_qlib_features`) |
| `rdagent/oai/` | LiteLLM backend (point at Claude here) |

## Appendix B — Config slots to override (env prefix `QLIB_QUANT_`)

`scen`, `quant_hypothesis_gen`, `factor_hypothesis2experiment`, `model_hypothesis2experiment`,
`factor_coder`, `model_coder`, `factor_runner`, `model_runner`, **`factor_summarizer`**,
**`model_summarizer`**, `action_selection` (`bandit|llm|random`),
`train_start/end`, `valid_start/end`, `test_start/end`, `evolving_n`.

## Appendix C — Run commands

```bash
# reproduce baseline (factor+model joint loop)
rdagent fin_quant
# factor-only / model-only
rdagent fin_factor
rdagent fin_model
# resume a session
rdagent fin_quant --path <LOG_PATH>/__session__/<i>/<step> --step_n 1
# view traces
rdagent ui            # (server_ui for the newer frontend)
# environment sanity
rdagent health_check
```

---

*Prepared from a direct read of the RD-Agent source. Claims about the guardrail gap, the pluggable
config, the loop structure, Docker execution, and the file map were verified against the code;
items flagged "verify" (exact LiteLLM/embedding env keys) were not run and should be confirmed in
the fork.*
