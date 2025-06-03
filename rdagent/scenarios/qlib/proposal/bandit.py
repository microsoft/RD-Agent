import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np


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


def extract_metrics_from_experiment(experiment) -> Metrics:
    """Extract metrics from experiment feedback"""
    try:
        result = experiment.result
        ic = result.get("IC", 0.0)
        icir = result.get("ICIR", 0.0)
        rank_ic = result.get("Rank IC", 0.0)
        rank_icir = result.get("Rank ICIR", 0.0)
        arr = result.get("1day.excess_return_with_cost.annualized_return ", 0.0)
        ir = result.get("1day.excess_return_with_cost.information_ratio", 0.0)
        mdd = result.get("1day.excess_return_with_cost.max_drawdown", 1.0)  # Avoid division by zero
        sharpe = arr / -mdd if mdd != 0 else 0.0

        return Metrics(ic=ic, icir=icir, rank_ic=rank_ic, rank_icir=rank_icir, arr=arr, ir=ir, mdd=mdd, sharpe=sharpe)
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return Metrics()


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
