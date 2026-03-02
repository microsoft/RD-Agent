"""Benchmark processors for core metric extraction.

Each benchmark has its own processor that knows how to extract
the core metric name and value from accuracy_summary data.
"""

from .bioprobench import BioProBenchProcessor
from .chemcotbench import ChemCotBenchProcessor
from .financeiq import FinanceIQProcessor
from .panorama import PanoramaProcessor
from .tablebench import TableBenchProcessor

PROCESSORS = [
    FinanceIQProcessor,
    PanoramaProcessor,
    ChemCotBenchProcessor,
    TableBenchProcessor,
    BioProBenchProcessor,
]


def get_core_metric_score(benchmark_name: str, accuracy_summary: dict) -> tuple[str, float, bool] | None:
    """Get core metric name, score, and direction for a benchmark.

    Args:
        benchmark_name: The benchmark name (e.g., "FinanceIQ", "panorama_par4pc")
        accuracy_summary: {dataset_name: {metric: value, ...}, ...}

    Returns:
        (metric_name, value, higher_is_better) or None
        - metric_name: includes "(average)" suffix if multiple datasets are averaged
        - value: the score
        - higher_is_better: True if higher values are better (use ↑), False otherwise (use ↓)
    """
    for processor in PROCESSORS:
        if processor.match(benchmark_name):
            return processor.get_core_metric(accuracy_summary)

    # Default fallback: use first numeric value with "accuracy" label
    scores = []
    for ds, metrics in accuracy_summary.items():
        if not isinstance(metrics, dict):
            continue
        if "accuracy" in metrics:
            scores.append(float(metrics["accuracy"]))
        else:
            for v in metrics.values():
                if isinstance(v, (int, float)):
                    scores.append(float(v))
                    break

    if not scores:
        return None

    avg = sum(scores) / len(scores)
    if len(scores) == 1:
        return ("accuracy", avg, True)  # higher is better
    else:
        return ("accuracy (average)", avg, True)  # higher is better


__all__ = [
    "get_core_metric_score",
    "PROCESSORS",
    "FinanceIQProcessor",
    "PanoramaProcessor",
    "ChemCotBenchProcessor",
    "TableBenchProcessor",
    "BioProBenchProcessor",
]
