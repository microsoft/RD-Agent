"""ChemCotBench benchmark processor."""

from .base import BenchmarkProcessor


class ChemCotBenchProcessor(BenchmarkProcessor):
    """ChemCotBench: Chemistry reasoning with various subtasks."""

    CORE_METRICS = {
        # Molecular understanding
        "mol_und_fg_count": "accuracy",
        "mol_und_ring_count": "accuracy",
        "mol_und_murcko_scaffold": "scaffold_hard",
        "mol_und_ring_system_scaffold": "score",
        "mol_und_equivalence": "accuracy",
        # Molecular editing
        "mol_edit_add": "correct_rate",
        "mol_edit_delete": "correct_rate",
        "mol_edit_sub": "correct_rate",
        # Molecular optimization (prefix match)
        "mol_opt_": "success_rate",
        # Reaction tasks
        "reaction_fs": "exact_match",
        "reaction_retro": "exact_match",
        "reaction_nepp": "exact_match",
        "reaction_rcr": "exact_match",
        "reaction_mechsel": "accuracy",
    }

    @classmethod
    def match(cls, benchmark_name: str) -> bool:
        return "chemcot" in benchmark_name.lower()

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
                # Prefix match for patterns ending with _
                if pattern.endswith("_"):
                    if pattern in ds_lower:
                        core_metric = metric
                        break
                else:
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
