"""
Compare ALPHA20 vs ALPHA158 baseline backtest performance.

This script runs backtests for both factor sets and compares key metrics:
- 年化收益 (Annualized Return)
- 最大回撤 (Maximum Drawdown)
- 信息比率 (Information Ratio)
- IC均值 (Mean IC)
- ICIR (IC Information Ratio)

Usage:
    python compare_alpha_baselines.py

Note: Requires Docker environment with qlib image prepared.
"""

from __future__ import annotations

import sys
from typing import Any

import docker
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.utils.qlib import ALPHA20, ALPHA158

BACKTEST_CONFIG = {
    "train_start": "2024-01-01",
    "train_end": "2024-12-31",
    "valid_start": "2025-01-01",
    "valid_end": "2025-06-30",
    "test_start": "2025-07-01",
    "test_end": "2026-03-30",
    "market": "csi500",
}


def create_experiment(
    factor_dict: dict[str, str],
    name: str,
) -> QlibFactorExperiment:
    """
    Create a QlibFactorExperiment with given factors.

    Args:
        factor_dict: Dictionary of factor names to expressions
        name: Experiment name for logging

    Returns:
        QlibFactorExperiment instance ready for backtest
    """
    logger.info(f"Creating experiment: {name} with {len(factor_dict)} factors")

    exp = QlibFactorExperiment()
    exp.base_features = factor_dict.copy()

    return exp


def run_backtest(exp: QlibFactorExperiment) -> dict[str, Any]:
    """
    Run backtest for the experiment and extract metrics.

    Args:
        exp: QlibFactorExperiment instance

    Returns:
        Dictionary with backtest metrics
    """
    runner = QlibFactorRunner()

    try:
        result_exp = runner.develop(exp)
        if result_exp.result is not None:
            return extract_metrics(result_exp.result)
        logger.error(f"Backtest failed: {result_exp.stdout}")
    except (RuntimeError, ValueError, KeyError) as e:
        logger.error(f"Exception during backtest: {e}")
        return {"error": str(e)}
    else:
        return {"error": result_exp.stdout}


def extract_metrics(result: Any) -> dict[str, Any]:
    """
    Extract key metrics from backtest result.

    Args:
        result: Backtest result (pandas Series or DataFrame)

    Returns:
        Dictionary with extracted metrics
    """
    metrics = {}

    if hasattr(result, "index"):
        for key in result.index:
            if "annualized_return" in key.lower():
                metrics["年化收益"] = result[key]
            if "max_drawdown" in key.lower():
                metrics["最大回撤"] = result[key]
            if "information_ratio" in key.lower():
                metrics["信息比率"] = result[key]
            if key.lower() == "ic.mean" or "ic_mean" in key.lower():
                metrics["IC均值"] = result[key]
            if key.lower() == "ic.ir" or "icir" in key.lower():
                metrics["ICIR"] = result[key]

    return metrics


def format_percentage(value: Any) -> str:
    """Format value as percentage string."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def format_number(value: Any) -> str:
    """Format value as number string."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def print_comparison_table(
    alpha20_results: dict[str, Any],
    alpha158_results: dict[str, Any],
) -> None:
    """
    Print comparison table for ALPHA20 vs ALPHA158.

    Args:
        alpha20_results: Metrics from ALPHA20 backtest
        alpha158_results: Metrics from ALPHA158 backtest
    """
    print("\n" + "=" * 60)
    print("ALPHA20 vs ALPHA158 Baseline Comparison")
    print("=" * 60)

    print(f"| {'Metric':<15} | {'ALPHA20':<12} | {'ALPHA158':<12} |")
    print(f"|{'-' * 17}|{'-' * 14}|{'-' * 14}|")

    metrics_order = ["年化收益", "最大回撤", "信息比率", "IC均值", "ICIR"]

    for metric in metrics_order:
        a20_val = alpha20_results.get(metric)
        a158_val = alpha158_results.get(metric)

        if metric in ["年化收益", "最大回撤"]:
            a20_str = format_percentage(a20_val)
            a158_str = format_percentage(a158_val)
        else:
            a20_str = format_number(a20_val)
            a158_str = format_number(a158_val)

        print(f"| {metric:<15} | {a20_str:<12} | {a158_str:<12} |")

    print("=" * 60)
    print(f"\nFactor Count: ALPHA20 = {len(ALPHA20)}, ALPHA158 = {len(ALPHA158)}")


def main() -> None:
    """Main entry point for baseline comparison."""
    print("=" * 60)
    print("ALPHA Baseline Comparison Script")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Train period: {BACKTEST_CONFIG['train_start']} ~ {BACKTEST_CONFIG['train_end']}")
    print(f"  Valid period: {BACKTEST_CONFIG['valid_start']} ~ {BACKTEST_CONFIG['valid_end']}")
    print(f"  Test period:  {BACKTEST_CONFIG['test_start']} ~ {BACKTEST_CONFIG['test_end']}")
    print(f"  Market:       {BACKTEST_CONFIG['market']}")
    print(f"\nALPHA20 factors: {len(ALPHA20)}")
    print(f"ALPHA158 factors: {len(ALPHA158)}")

    try:
        client = docker.from_env()
        client.ping()
        print("\n✓ Docker connection successful")
    except (docker.errors.DockerException, OSError) as e:
        print(f"\n⚠ Docker not available: {e}")
        print("  Backtests require Docker with qlib image")
        print("  Run: docker build -t local_qlib:latest -f Dockerfile_qlib .")
        sys.exit(1)

    print("\n" + "-" * 60)
    print("Creating experiments...")
    alpha20_exp = create_experiment(ALPHA20, "ALPHA20")
    alpha158_exp = create_experiment(ALPHA158, "ALPHA158")

    print("\n" + "-" * 60)
    print("Running ALPHA20 backtest...")
    alpha20_results = run_backtest(alpha20_exp)

    print("\n" + "-" * 60)
    print("Running ALPHA158 backtest...")
    alpha158_results = run_backtest(alpha158_exp)

    print_comparison_table(alpha20_results, alpha158_results)


if __name__ == "__main__":
    main()
