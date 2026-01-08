"""Base class for benchmark core metric extraction."""

from abc import ABC, abstractmethod


class BenchmarkProcessor(ABC):
    """Base class for benchmark core metric extraction."""

    # Metrics where higher values are better (default assumption)
    # Override in subclass if needed
    HIGHER_IS_BETTER: set[str] = {
        "accuracy",
        "exact_match",
        "f1",
        "f1_score",
        "macro_f1",
        "correct_rate",
        "success_rate",
        "gold_hit_rate",
        "score",
        "scaffold_hard",
        "kendall_tau",
        "ROUGE-L",
    }

    @classmethod
    @abstractmethod
    def match(cls, benchmark_name: str) -> bool:
        """Check if this processor handles the given benchmark."""
        pass

    @classmethod
    @abstractmethod
    def get_core_metric(cls, accuracy_summary: dict) -> tuple[str, float, bool] | None:
        """Extract core metric name, value, and direction from accuracy_summary.

        Args:
            accuracy_summary: {dataset_name: {metric: value, ...}, ...}

        Returns:
            (metric_name, value, higher_is_better) or None
            - metric_name: includes "(average)" suffix if multiple datasets
            - value: the score
            - higher_is_better: True if higher values are better, False otherwise
        """
        pass

    @classmethod
    def is_higher_better(cls, metric_name: str) -> bool:
        """Check if higher values are better for this metric."""
        # Remove (average) suffix for checking
        base_metric = metric_name.replace(" (average)", "").strip()
        return base_metric.lower() in {m.lower() for m in cls.HIGHER_IS_BETTER}
