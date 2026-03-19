"""
AutoRL-Bench Metrics

Calculate run-level process metrics and generate visualizations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rdagent.scenarios.rl.autorl_bench.core.utils import read_run_meta


def _parse_iso_time(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_metrics(
    workspace: Path,
    baseline: Optional[float],
    base_model_path: Optional[str],
) -> dict[str, Any]:
    scores_file = workspace / "scores.json"
    scores = json.loads(scores_file.read_text()) if scores_file.exists() else []
    score_values = [entry.get("score", 0.0) for entry in scores]

    valid_scores = [s for s in score_values if s and s > 0]
    valid_submission_rate = _safe_div(len(valid_scores), len(score_values))

    first_valid_idx = None
    for idx, score in enumerate(score_values, start=1):
        if score and score > 0:
            first_valid_idx = idx
            break

    run_meta = read_run_meta(workspace)
    start_time = run_meta.get("start_time")
    end_time = run_meta.get("end_time")
    timeout_s = run_meta.get("timeout_s")
    last_submit_time = run_meta.get("last_submit_time")

    first_valid_delay = None
    if start_time and first_valid_idx:
        first_time = _parse_iso_time(scores[first_valid_idx - 1].get("timestamp"))
        if first_time:
            first_valid_delay = int(first_time.timestamp() - start_time)

    time_to_first_improvement = None
    if baseline is not None and start_time:
        for entry in scores:
            if entry.get("score", 0.0) > baseline:
                ts = _parse_iso_time(entry.get("timestamp"))
                if ts:
                    time_to_first_improvement = int(ts.timestamp() - start_time)
                break

    monotonic_ratio = None
    if len(score_values) >= 2:
        increases = sum(1 for prev, cur in zip(score_values, score_values[1:]) if cur > prev)
        monotonic_ratio = _safe_div(increases, len(score_values) - 1)

    copy_model_count = None
    if base_model_path:
        base_path = Path(base_model_path).resolve()
        copy_model_count = 0
        for entry in scores:
            model_path = entry.get("model_path")
            if model_path and Path(model_path).resolve() == base_path:
                copy_model_count += 1

    time_used_ratio = None
    if start_time and end_time and timeout_s:
        time_used_ratio = _safe_div(end_time - start_time, timeout_s)

    time_to_best = None
    if start_time and scores:
        best_entry = max(scores, key=lambda x: x.get("score", 0.0))
        best_ts = _parse_iso_time(best_entry.get("timestamp"))
        if best_ts:
            time_to_best = int(best_ts.timestamp() - start_time)

    last_submit_gap = None
    if end_time and last_submit_time:
        last_submit_gap = int(end_time - last_submit_time)

    return {
        "valid_submission_rate": valid_submission_rate,
        "first_valid_idx": first_valid_idx,
        "first_valid_delay": first_valid_delay,
        "score_trajectory": score_values,
        "time_to_first_improvement": time_to_first_improvement,
        "time_to_best": time_to_best,
        "monotonic_ratio": monotonic_ratio,
        "copy_model_count": copy_model_count,
        "time_used_ratio": time_used_ratio,
        "last_submit_gap": last_submit_gap,
    }


def write_metrics_json(workspace: Path, metrics: dict[str, Any]) -> Path:
    reports_dir = workspace / "reports"
    reports_dir.mkdir(exist_ok=True)
    target = reports_dir / "metrics.json"
    target.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return target


def plot_score_trajectory(workspace: Path, metrics: dict[str, Any]) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    scores = metrics.get("score_trajectory", [])
    reports_dir = workspace / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    target = figures_dir / "score_trajectory.png"

    plt.figure(figsize=(6, 4))
    if scores:
        plt.plot(list(range(1, len(scores) + 1)), scores, marker="o")
        plt.xlabel("submission")
        plt.ylabel("score")
        plt.title("score trajectory")
    else:
        plt.text(0.5, 0.5, "no submissions", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def run_workspace_metrics(
    workspace: Path,
    baseline: Optional[float],
    base_model_path: Optional[str],
    *,
    plot: bool = True,
) -> dict[str, Any]:
    metrics = compute_metrics(workspace, baseline, base_model_path)
    write_metrics_json(workspace, metrics)
    if plot:
        plot_score_trajectory(workspace, metrics)
    return metrics
