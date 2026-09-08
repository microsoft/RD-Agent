import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.metrics import (
    ARR_KEY,
    IC_KEY,
    ICIR_KEY,
    IR_KEY,
    MDD_KEY,
    RANK_IC_KEY,
    RANK_ICIR_KEY,
)


@dataclass
class Metrics:
    ic: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    rank_icir: float = 0.0
    arr: float = 0.0
    ir: float = 0.0
    mdd: float = 0.0
    sharpe: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.ic,
                self.icir,
                self.rank_ic,
                self.rank_icir,
                self.arr,
                self.ir,
                -self.mdd,
                self.sharpe,
            ]
        )


def _get_metric(result, key: str, default: float = 0.0) -> float:
    """Read one metric from ``experiment.result``, warning instead of silently defaulting when it is absent.

    A missing key, a mistyped key and a genuinely zero metric used to be indistinguishable (see #1451); the
    warning makes the first two visible in the log while keeping the loop alive on a partial Qlib result.
    """
    if key in result:
        return float(result[key])
    logger.warning(
        f"Metric {key!r} not found in experiment result, using {default}. Available keys: {list(result.keys())}"
    )
    return default


def extract_metrics_from_experiment(experiment) -> Metrics:
    """Extract the bandit's feature vector from ``experiment.result`` (a Series indexed by Qlib metric name)."""
    result = getattr(experiment, "result", None)
    if result is None:
        # Execution failed, so there is nothing to learn from; a zero vector is neutral for the bandit.
        logger.warning("Experiment has no result, using all-zero metrics for the bandit")
        return Metrics()

    ic = _get_metric(result, IC_KEY)
    icir = _get_metric(result, ICIR_KEY)
    rank_ic = _get_metric(result, RANK_IC_KEY)
    rank_icir = _get_metric(result, RANK_ICIR_KEY)
    arr = _get_metric(result, ARR_KEY)
    ir = _get_metric(result, IR_KEY)
    # Qlib reports max drawdown as a number <= 0. A default of 0.0 (guarded below) keeps both the ratio and the
    # -mdd vector slot at zero when the key is missing; the previous default of 1.0 flipped the ratio's sign.
    mdd = _get_metric(result, MDD_KEY)
    sharpe = arr / -mdd if mdd != 0 else 0.0

    return Metrics(ic=ic, icir=icir, rank_ic=rank_ic, rank_icir=rank_icir, arr=arr, ir=ir, mdd=mdd, sharpe=sharpe)


class LinearThompsonTwoArm:
    def __init__(self, dim: int, prior_var: float = 1.0, noise_var: float = 1.0):
        self.dim = dim
        self.noise_var = noise_var
        # Each arm has its own posterior: mean & inverse of covariance (precision matrix)
        self.mean = {
            "factor": np.zeros(dim),
            "model": np.zeros(dim),
        }
        self.precision = {
            "factor": np.eye(dim) / prior_var,
            "model": np.eye(dim) / prior_var,
        }

    def sample_reward(self, arm: str, x: np.ndarray) -> float:
        P = self.precision[arm]
        P = 0.5 * (P + P.T)

        eps = 1e-6
        try:
            cov = np.linalg.inv(P + eps * np.eye(self.dim))
            L = np.linalg.cholesky(cov)
            z = np.random.randn(self.dim)
            w_sample = self.mean[arm] + L @ z
        except np.linalg.LinAlgError:
            w_sample = self.mean[arm]

        return float(np.dot(w_sample, x))

    def update(self, arm: str, x: np.ndarray, r: float) -> None:
        P = self.precision[arm]
        P += np.outer(x, x) / self.noise_var
        self.precision[arm] = P
        self.mean[arm] = np.linalg.solve(P, P @ self.mean[arm] + (r / self.noise_var) * x)

    def next_arm(self, x: np.ndarray) -> str:
        scores = {arm: self.sample_reward(arm, x) for arm in ("factor", "model")}
        return max(scores, key=scores.get)


class EnvController:
    def __init__(self, weights: Tuple[float, ...] = None) -> None:
        self.weights = np.asarray(weights or (0.1, 0.1, 0.05, 0.05, 0.25, 0.15, 0.1, 0.2))
        self.bandit = LinearThompsonTwoArm(dim=8, prior_var=10.0, noise_var=0.5)

    def reward(self, m: Metrics) -> float:
        return float(np.dot(self.weights, m.as_vector()))

    def decide(self, m: Metrics) -> str:
        x = m.as_vector()
        return self.bandit.next_arm(x)

    def record(self, m: Metrics, arm: str) -> None:
        r = self.reward(m)
        self.bandit.update(arm, m.as_vector(), r)
