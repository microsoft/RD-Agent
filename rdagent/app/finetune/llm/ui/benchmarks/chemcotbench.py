"""ChemCotBench benchmark processor."""

from .base import BenchmarkProcessor


class ChemCotBenchProcessor(BenchmarkProcessor):
    """ChemCotBench: Chemistry reasoning with various subtasks.

    All metrics are 0-100 percentages, enabling unified averaging within each subset.
    """

    # Define core metric field names for each task
    CORE_METRICS = {
        # Molecular understanding
        "mol_und_fg_count": "accuracy",
        "mol_und_ring_count": "accuracy",
        "mol_und_murcko_scaffold": "scaffold_hard",  # Exact match rate (0-100%)
        "mol_und_ring_system_scaffold": "score",  # "Yes" ratio (0-100%)
        "mol_und_equivalence": "accuracy",
        # Molecular editing
        "mol_edit_add": "correct_rate",
        "mol_edit_delete": "correct_rate",
        "mol_edit_sub": "correct_rate",
        # Molecular optimization (prefix match)
        "mol_opt_": "success_rate",
        # Reaction tasks - unified to exact_match
        "reaction_fs": "exact_match",
        "reaction_retro": "exact_match",
        "reaction_nepp": "exact_match",
        "reaction_rcr": "exact_match",
        "reaction_mechsel": "exact_match",  # Will fallback to accuracy if exact_match not found
    }

    # Metric groups: unified display names for each subset
    METRIC_GROUPS = {
        "mol_und": "accuracy",  # mol_und subset displays as accuracy
        "mol_edit": "correct_rate",
        "mol_opt": "success_rate",
        "reaction": "exact_match",  # reaction subset displays as exact_match
    }

    @classmethod
    def match(cls, benchmark_name: str) -> bool:
        return "chemcot" in benchmark_name.lower()

    @classmethod
    def get_core_metric(cls, accuracy_summary: dict) -> tuple[str, float, bool] | None:
        scores = []
        group_detected = None

        for ds, metrics in accuracy_summary.items():
            if not isinstance(metrics, dict):
                continue
            ds_lower = ds.lower()

            # Detect subset type
            for group in cls.METRIC_GROUPS:
                if group in ds_lower:
                    group_detected = group
                    break

            # Find matching core metric
            core_metric = "accuracy"  # fallback
            for pattern, metric in cls.CORE_METRICS.items():
                # Prefix match for patterns ending with _
                if pattern.endswith("_"):
                    if pattern in ds_lower:
                        core_metric = metric
                        break
                else:
                    if pattern in ds_lower:
                        core_metric = metric
                        break

            # Try to get metric value with fallback support
            value = None
            if core_metric in metrics:
                value = float(metrics[core_metric])
            elif core_metric == "exact_match" and "accuracy" in metrics:
                # reaction_mechsel fallback: exact_match -> accuracy
                value = float(metrics["accuracy"])

            if value is not None:
                scores.append(value)

        if not scores:
            return None

        avg = sum(scores) / len(scores)

        # Use unified metric name for the detected subset
        if group_detected and group_detected in cls.METRIC_GROUPS:
            unified_name = cls.METRIC_GROUPS[group_detected]
            if len(scores) == 1:
                metric_name = unified_name
            else:
                metric_name = f"{unified_name} (average)"
        else:
            # Fallback for unknown subsets
            if len(scores) == 1:
                metric_name = "accuracy"
            else:
                metric_name = "accuracy (average)"

        return (metric_name, avg, cls.is_higher_better(metric_name))
