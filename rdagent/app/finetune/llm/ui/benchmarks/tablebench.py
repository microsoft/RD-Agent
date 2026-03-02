"""TableBench benchmark processor."""

from .base import BenchmarkProcessor


class TableBenchProcessor(BenchmarkProcessor):
    """TableBench: Table QA with different subtasks."""

    CORE_METRICS = {
        "fact": "accuracy",
        "numerical": "accuracy",
        "analysis": "accuracy",
        "visualization": "Pass@1",  # TableBench visualization uses Pass@1 as core metric
    }

    # TableBench-specific metrics where higher is better
    HIGHER_IS_BETTER = BenchmarkProcessor.HIGHER_IS_BETTER | {
        "Pass@1",
        "ECR@1",
        "Parse@1",
    }

    @classmethod
    def match(cls, benchmark_name: str) -> bool:
        return "tablebench" in benchmark_name.lower()

    @classmethod
    def get_core_metric(cls, accuracy_summary: dict) -> tuple[str, float, bool] | None:
        scores = []
        metrics_used = []

        for ds, metrics in accuracy_summary.items():
            if not isinstance(metrics, dict):
                continue
            ds_lower = ds.lower()
            # Find matching core metric
            core_metric = "accuracy"  # fallback
            for pattern, metric in cls.CORE_METRICS.items():
                if pattern in ds_lower:
                    core_metric = metric
                    break

            if core_metric in metrics:
                scores.append(float(metrics[core_metric]))
                metrics_used.append(core_metric)

        if not scores:
            return None

        avg = sum(scores) / len(scores)
        unique = list(set(metrics_used))

        if len(scores) == 1:
            metric_name = unique[0]
        elif len(unique) == 1:
            metric_name = f"{unique[0]} (average)"
        else:
            metric_name = "mixed (average)"

        return (metric_name, avg, cls.is_higher_better(metric_name))
