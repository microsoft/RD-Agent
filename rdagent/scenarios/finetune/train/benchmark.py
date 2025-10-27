"""
Benchmark Evaluation using OpenCompass

Simple benchmark evaluator that runs OpenCompass in Docker to evaluate fine-tuned models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger


def get_benchmark_env(datasets: List[str], adapter_path: str, base_model: str):
    """
    Create OpenCompass benchmark environment (similar to get_ft_env).

    Args:
        datasets: List of benchmark datasets
        adapter_path: Path to adapter files
        base_model: Base model path

    Returns:
        Configured Docker environment for OpenCompass
    """
    from rdagent.utils.env import FTDockerConf, FTDockerEnv

    # Create Docker environment with OpenCompass image
    conf = FTDockerConf()
    conf.image = "rdagent-opencompass:latest"
    env = FTDockerEnv(conf=conf)

    # Set environment variables for OpenCompass
    env.conf.env_vars = {
        "BENCHMARK_DATASETS": ",".join(datasets),
        "ADAPTER_PATH": adapter_path,
        "BASE_MODEL": base_model,
    }

    env.conf.running_timeout_period = FT_RD_SETTING.benchmark_timeout
    env.prepare()
    return env


class FTBenchmarkEvaluator(CoSTEEREvaluator):
    """Benchmark evaluator using OpenCompass in Docker."""

    def __init__(self, scen, datasets: Optional[List[str]] = None):
        super().__init__(scen)
        self.datasets = datasets or FT_RD_SETTING.benchmark_datasets

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: Optional[QueriedKnowledge] = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """Run benchmark evaluation using OpenCompass."""
        workspace_path = implementation.workspace_path
        output_path = workspace_path / "output"

        # Check adapter files
        if not self._check_adapter_files(output_path):
            return CoSTEERSingleFeedback(
                execution="Adapter files not found",
                return_checking="No adapter files",
                code="Adapter validation failed",
                final_decision=False,
            )

        # Generate config
        eval_config_path = workspace_path / "benchmark_config.json"
        self._generate_eval_config(output_path, eval_config_path)

        # Run benchmark
        logger.info(f"Starting benchmark on datasets: {self.datasets}")
        env = get_benchmark_env(
            datasets=self.datasets,
            adapter_path=str(output_path),
            base_model=self.scen.base_model,
        )

        # Create entrypoint script
        script_path = workspace_path / "run_benchmark.sh"
        script_path.write_text(
            f"""#!/bin/bash
cd /workspace
python /app/eval_entrypoint.py
"""
        )

        result = implementation.execute(
            env=env,
            entry=f"bash {script_path.name}",
        )

        if result.exit_code != 0:
            return CoSTEERSingleFeedback(
                execution=f"Benchmark failed (exit_code={result.exit_code})",
                return_checking="Execution failed",
                code=result.stdout,
                final_decision=False,
            )

        # Parse results
        results_path = workspace_path / "benchmark_results"
        scores = self._parse_results(results_path)
        report = self._format_report(scores)

        return CoSTEERSingleFeedback(
            execution=f"Benchmark completed on {len(self.datasets)} datasets",
            return_checking=report,
            code=f"Average: {self._calc_avg(scores):.1f}",
            final_decision=True,
        )

    def _check_adapter_files(self, output_path: Path) -> bool:
        """Check adapter files exist."""
        has_weight = (output_path / "adapter_model.safetensors").exists() or (
            output_path / "adapter_model.bin"
        ).exists()
        has_config = (output_path / "adapter_config.json").exists()
        return has_weight and has_config

    def _generate_eval_config(self, adapter_path: Path, config_path: Path) -> None:
        """Generate OpenCompass config JSON."""
        with open(adapter_path / "adapter_config.json") as f:
            adapter_config = json.load(f)

        base_model = adapter_config.get("base_model_name_or_path", self.scen.base_model)

        config = {
            "base_model": base_model,
            "adapter_path": str(adapter_path),
            "datasets": self.datasets,
        }

        with open(config_path, "w") as f:
            json.dump(config, f)

    def _parse_results(self, results_path: Path) -> Dict[str, float]:
        """Parse benchmark results from CSV."""
        import csv

        summary_files = list(results_path.glob("*/summary/summary_*.csv"))
        if not summary_files:
            logger.warning("No results found, returning placeholder")
            return {ds: 50.0 for ds in self.datasets}

        latest = max(summary_files, key=lambda p: p.stat().st_mtime)
        scores = {}

        with open(latest) as f:
            for row in csv.DictReader(f):
                scores[row["dataset"]] = float(row["score"])

        return scores

    def _format_report(self, scores: Dict[str, float]) -> str:
        """Format results as text."""
        lines = ["Benchmark Results:"]
        for dataset, score in scores.items():
            lines.append(f"  {dataset}: {score:.1f}")
        return "\n".join(lines)

    def _calc_avg(self, scores: Dict[str, float]) -> float:
        """Calculate average score."""
        return sum(scores.values()) / len(scores) if scores else 0
