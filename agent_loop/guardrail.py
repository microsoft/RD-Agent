"""
rdagent-guardrail — the deterministic promote/reject gate (ADR 0002 Phase B; ADR 0001 §7 v0).

This is the scientific crux of the agent-driven loop: it replaces RD-Agent's LLM-judge summarizer
(which lets an LLM eyeball test-segment metrics — textbook selection-on-the-test-set) with a
*deterministic* rule. There is NO LLM here.

A candidate is promoted to SOTA only if ALL gates pass:

  1. holdout_ok      — beats the current SOTA on a LOCKED holdout segment the loop never selects on
                       (or clears an absolute floor if there is no incumbent).
  2. dsr_ok          — its Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) exceeds a threshold.
                       DSR haircuts the Sharpe for (a) the number of trials tried so far and (b) the
                       non-normality (skew/kurtosis) of returns. As trial count grows, the bar rises,
                       so marginal "winners" found by searching get rejected.
  3. net_positive    — its net-of-cost information ratio is positive (reject pure gross winners).
  4. beats_sota_net  — its net-of-cost IR beats the incumbent's by a margin.

Inputs are the JSON emitted by `agent_loop.run_qlib` (which records the workspace path, so the raw
daily-return series in `<workspace>/ret.pkl` is available for honest T / skew / kurtosis).

Trace JSON schema (shared with the Phase C `trace` tool) — all fields optional except `trials`:
    {"trials": [{"loop": int, "action": str, "decision": bool,
                 "sr_period": float, "holdout_metrics": {metric: value}, ...}],
     "sota_loop": int | null}
N (number of trials for the DSR haircut) = len(trials) + 1 (this candidate).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329
TRADING_DAYS = 252


# --------------------------------------------------------------------------- stats
def daily_excess_returns(workspace: str | Path, with_cost: bool = True) -> pd.Series:
    """Daily excess return series from a run's ret.pkl (qlib report_normal_1day).

    excess_with_cost = return - bench - cost ;  excess_without_cost = return - bench.
    """
    ret_path = Path(workspace) / "ret.pkl"
    if not ret_path.exists():
        raise FileNotFoundError(f"ret.pkl not found in workspace: {ret_path}")
    df = pd.read_pickle(ret_path)
    exc = df["return"] - df["bench"]
    if with_cost:
        exc = exc - df["cost"]
    return exc.dropna()


def sharpe_stats(returns: pd.Series) -> dict[str, float]:
    """Per-period Sharpe plus the higher moments the DSR standard error needs."""
    r = np.asarray(returns, dtype=float)
    T = int(r.size)
    mean, std = float(r.mean()), float(r.std(ddof=1))
    sr = mean / std if std > 0 else 0.0
    return {
        "T": T,
        "mean": mean,
        "std": std,
        "sr_period": sr,
        "sr_ann": sr * math.sqrt(TRADING_DAYS),
        "skew": float(pd.Series(r).skew()),
        "kurt": float(pd.Series(r).kurt() + 3.0),  # pandas kurt is excess; DSR wants raw (normal=3)
    }


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, T: int, skew: float, kurt: float) -> float:
    """PSR: P(true per-period SR > sr_benchmark), using the Mertens standard error of Sharpe."""
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    denom = max(denom, 1e-12)
    z = (sr - sr_benchmark) * math.sqrt(max(T - 1, 1)) / math.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr_period: float) -> float:
    """Expected maximum per-period Sharpe under the null of `n_trials` noise strategies."""
    if n_trials <= 1 or var_sr_period <= 0:
        return 0.0
    sigma = math.sqrt(var_sr_period)
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sigma * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe_ratio(
    stats: dict[str, float], n_trials: int, var_sr_period: float | None
) -> dict[str, Any]:
    """DSR = PSR against the expected-max-Sharpe benchmark implied by multiple testing."""
    sr, T, skew, kurt = stats["sr_period"], stats["T"], stats["skew"], stats["kurt"]
    if var_sr_period is None:
        # No trial-dispersion estimate yet: assume trials are null noise, Var(SR)~1/T.
        var_sr_period = 1.0 / max(T - 1, 1)
        var_source = "default_1/T"
    else:
        var_source = "trace_trial_variance"
    sr0 = expected_max_sharpe(n_trials, var_sr_period)
    dsr = probabilistic_sharpe_ratio(sr, sr0, T, skew, kurt)
    return {
        "value": dsr,
        "n_trials": n_trials,
        "benchmark_sr_period": sr0,
        "benchmark_sr_ann": sr0 * math.sqrt(TRADING_DAYS),
        "var_sr_period": var_sr_period,
        "var_sr_source": var_source,
    }


# --------------------------------------------------------------------------- helpers
def _load(obj: str | dict | None) -> dict | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    return json.loads(Path(obj).read_text())


def _ir_with_cost(res: dict) -> float:
    return float(res["metrics"]["1day.excess_return_with_cost.information_ratio"])


def _sota_from_trace(trace: dict | None) -> dict | None:
    if not trace or not trace.get("trials"):
        return None
    accepted = [t for t in trace["trials"] if t.get("decision")]
    if not accepted:
        return None
    sota_loop = trace.get("sota_loop")
    if sota_loop is not None:
        for t in accepted:
            if t.get("loop") == sota_loop:
                return t
    return accepted[-1]


# --------------------------------------------------------------------------- gate
def evaluate(
    candidate: str,
    holdout: str | None = None,
    trace: str | None = None,
    sota: str | None = None,
    dsr_threshold: float = 0.95,
    cost_margin: float = 0.0,
    holdout_metric: str = "Rank IC",
    min_holdout: float = 0.0,
    out: str | None = None,
) -> dict[str, Any]:
    """Decide whether `candidate` should replace the current SOTA. Returns a decision dict.

    candidate/holdout/sota: paths to run_qlib JSON (selection segment, locked-holdout segment, and
    the incumbent's selection JSON, respectively). trace: path to the trace ledger JSON.
    """
    cand = _load(candidate)
    hold = _load(holdout)
    trace_d = _load(trace)
    sota_sel = _load(sota) or _sota_from_trace(trace_d)

    stats = sharpe_stats(daily_excess_returns(cand["workspace"], with_cost=True))
    n_trials = (len(trace_d["trials"]) + 1) if (trace_d and trace_d.get("trials")) else 1

    var_sr = None
    if trace_d and trace_d.get("trials"):
        srs = [t["sr_period"] for t in trace_d["trials"] if t.get("sr_period") is not None]
        if len(srs) >= 2:
            var_sr = float(np.var(srs, ddof=1))
    dsr = deflated_sharpe_ratio(stats, n_trials, var_sr)

    reasons: list[str] = []

    # gate: DSR
    dsr_ok = dsr["value"] >= dsr_threshold
    reasons.append(
        f"DSR={dsr['value']:.4f} {'>=' if dsr_ok else '<'} {dsr_threshold} "
        f"(N={n_trials}, benchmark SR_ann={dsr['benchmark_sr_ann']:.3f}, var_src={dsr['var_sr_source']})"
    )

    # gate: net-of-cost positive
    cand_ir_net = _ir_with_cost(cand)
    net_positive = cand_ir_net > 0.0
    reasons.append(f"net IR(with cost)={cand_ir_net:+.4f} {'> 0' if net_positive else '<= 0 (reject gross-only)'}")

    # gate: beats SOTA net-of-cost
    sota_ir_net = _ir_with_cost(sota_sel) if sota_sel and "metrics" in sota_sel else 0.0
    beats_sota_net = cand_ir_net > sota_ir_net + cost_margin
    reasons.append(
        f"net IR beats SOTA: {cand_ir_net:+.4f} vs {sota_ir_net:+.4f}+{cost_margin} -> {beats_sota_net}"
        + ("" if sota_sel else " (no incumbent -> bar is 0)")
    )

    # gate: locked holdout
    if hold is None:
        holdout_ok = False
        reasons.append("HOLDOUT MISSING -> cannot promote (a locked-holdout run is required)")
    else:
        cand_h = float(hold["metrics"].get(holdout_metric))
        sota_h = None
        if sota_sel and sota_sel.get("holdout_metrics"):
            sota_h = sota_sel["holdout_metrics"].get(holdout_metric)
        bar = sota_h if sota_h is not None else min_holdout
        holdout_ok = cand_h > bar
        reasons.append(
            f"holdout {holdout_metric}={cand_h:+.4f} {'>' if holdout_ok else '<='} "
            f"{bar:+.4f} ({'SOTA holdout' if sota_h is not None else 'floor'})"
        )

    decision = bool(holdout_ok and dsr_ok and net_positive and beats_sota_net)

    result = {
        "decision": decision,
        "candidate_workspace": cand["workspace"],
        "n_trials": n_trials,
        "candidate": {
            "sr_period": stats["sr_period"],
            "sr_ann": stats["sr_ann"],
            "T": stats["T"],
            "skew": stats["skew"],
            "kurt": stats["kurt"],
            "ir_with_cost": cand_ir_net,
        },
        "dsr": dsr,
        "gates": {
            "holdout_ok": holdout_ok,
            "dsr_ok": dsr_ok,
            "net_positive": net_positive,
            "beats_sota_net": beats_sota_net,
        },
        "reasons": reasons,
    }

    print(f"\n==== guardrail: {'PROMOTE' if decision else 'REJECT'} ====")
    for r in reasons:
        print(" -", r)
    if out is not None:
        Path(out).write_text(json.dumps(result, indent=2))
        print(f"\n[guardrail] wrote {out}")
    return result


if __name__ == "__main__":
    fire.Fire(evaluate)
