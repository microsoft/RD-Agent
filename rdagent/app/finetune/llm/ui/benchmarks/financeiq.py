"""FinanceIQ benchmark processor."""

from .base import BenchmarkProcessor


class FinanceIQProcessor(BenchmarkProcessor):
    """FinanceIQ: 10 exam subjects, all use accuracy."""

    @classmethod
    def match(cls, benchmark_name: str) -> bool:
        return "financeiq" in benchmark_name.lower()

    @classmethod
    def get_core_metric(cls, accuracy_summary: dict) -> tuple[str, float, bool] | None:
        scores = []
        for ds, metrics in accuracy_summary.items():
            if not isinstance(metrics, dict):
                continue
            if "accuracy" in metrics:
                scores.append(float(metrics["accuracy"]))

        if not scores:
            return None

        avg = sum(scores) / len(scores)
        if len(scores) == 1:
            return ("accuracy", avg, True)  # higher is better
        else:
            return ("accuracy (average)", avg, True)  # higher is better
