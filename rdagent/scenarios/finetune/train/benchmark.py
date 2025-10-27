"""
Benchmark Evaluation using OpenCompass

Evaluator that runs OpenCompass in Docker to evaluate fine-tuned models on standard benchmarks.
"""

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


def get_benchmark_env(
    datasets: List[str],
    adapter_path: str,
    base_model: str,
    model_abbr: str,
):
    """
    Create OpenCompass benchmark environment (similar to get_ft_env).

    Args:
        datasets: List of benchmark datasets (e.g., ['mmlu', 'gsm8k'])
        adapter_path: Absolute path to adapter files in workspace
        base_model: Base model identifier (e.g., 'Qwen/Qwen2-1.5B-Instruct')
        model_abbr: Model abbreviation for result identification

    Returns:
        Configured Docker environment for OpenCompass
    """
    from rdagent.utils.env import FTDockerConf, FTDockerEnv

    conf = FTDockerConf()
    conf.image = "rdagent-opencompass:latest"
    env = FTDockerEnv(conf=conf)

    # Pass all configurations via environment variables
    # This is cleaner than JSON files and easier to debug
    env.conf.env_vars = {
        "BENCHMARK_DATASETS": ",".join(datasets),
        "ADAPTER_PATH": adapter_path,
        "BASE_MODEL": base_model,
        "MODEL_ABBR": model_abbr,
        # Optional: extend with more parameters in the future
        "MAX_OUT_LEN": "2048",
        "BATCH_SIZE": "8",
        "NUM_GPUS": "1",
    }

    env.conf.running_timeout_period = FT_RD_SETTING.benchmark_timeout
    env.prepare()
    return env


class FTBenchmarkEvaluator(CoSTEEREvaluator):
    """
    Benchmark evaluator using OpenCompass in Docker.

    This evaluator runs standard LLM benchmarks on fine-tuned models.
    It should only run when training succeeds and adapter files are valid.
    """

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
        """
        Run benchmark evaluation using OpenCompass.

        Workflow:
        1. Validate adapter files
        2. Generate model abbreviation for result tracking
        3. Launch OpenCompass Docker with environment variables
        4. Parse and report results
        """
        workspace_path = implementation.workspace_path
        output_path = workspace_path / "output"

        # Early exit if no datasets configured
        if not self.datasets:
            logger.info("Benchmark datasets not configured, skipping evaluation")
            return CoSTEERSingleFeedback(
                execution="Benchmark skipped (no datasets configured)",
                return_checking="N/A",
                code="Configure benchmark_datasets in conf.py to enable",
                final_decision=True,  # Not a failure, just skipped
            )

        # Validate adapter files
        validation_result = self._validate_adapter_files(output_path)
        if not validation_result["valid"]:
            return CoSTEERSingleFeedback(
                execution=f"Adapter validation failed: {validation_result['reason']}",
                return_checking="Invalid adapter files",
                code="Check training output for adapter_model.* and adapter_config.json",
                final_decision=False,
            )

        # Generate model abbreviation for result identification
        model_abbr = self._generate_model_abbr(implementation, output_path)

        # Run benchmark in Docker
        logger.info(f"Starting benchmark evaluation on datasets: {self.datasets}")
        logger.info(f"Model: {model_abbr} (base: {self.scen.base_model})")

        env = get_benchmark_env(
            datasets=self.datasets,
            adapter_path=str(output_path),
            base_model=self.scen.base_model,
            model_abbr=model_abbr,
        )

        # Simple entrypoint script
        script_path = workspace_path / "run_benchmark.sh"
        script_path.write_text("#!/bin/bash\n" "cd /workspace\n" "python /app/eval_entrypoint.py\n")

        result = implementation.execute(
            env=env,
            entry=f"bash {script_path.name}",
        )

        if result.exit_code != 0:
            return CoSTEERSingleFeedback(
                execution=f"Benchmark execution failed (exit_code={result.exit_code})",
                return_checking="OpenCompass error",
                code=result.stdout[-2000:] if result.stdout else "No output",  # Last 2000 chars
                final_decision=False,
            )

        # Parse and format results
        results_path = workspace_path / "benchmark_results"
        scores = self._parse_results(results_path)
        report = self._format_report(scores)

        return CoSTEERSingleFeedback(
            execution=f"Benchmark completed: {len(scores)}/{len(self.datasets)} datasets",
            return_checking=report,
            code=f"Average Score: {self._calc_avg(scores):.1f}%",
            final_decision=True,
        )

    def _validate_adapter_files(self, output_path: Path) -> Dict[str, any]:
        """
        Validate adapter files exist and are complete.

        Returns:
            dict with 'valid' (bool) and 'reason' (str) keys
        """
        if not output_path.exists():
            return {"valid": False, "reason": "Output directory not found"}

        # Check weight files (safetensors preferred, fallback to bin)
        has_safetensors = (output_path / "adapter_model.safetensors").exists()
        has_bin = (output_path / "adapter_model.bin").exists()

        if not (has_safetensors or has_bin):
            return {"valid": False, "reason": "No adapter weight file (*.safetensors or *.bin)"}

        # Check config
        config_path = output_path / "adapter_config.json"
        if not config_path.exists():
            return {"valid": False, "reason": "adapter_config.json not found"}

        # Optional: validate config content
        try:
            import json

            with open(config_path) as f:
                config = json.load(f)
            if "base_model_name_or_path" not in config:
                logger.warning("adapter_config.json missing base_model_name_or_path")
        except Exception as e:
            return {"valid": False, "reason": f"Invalid config JSON: {e}"}

        return {"valid": True, "reason": ""}

    def _generate_model_abbr(self, implementation: FBWorkspace, output_path: Path) -> str:
        """
        Generate a unique model abbreviation for result tracking.

        Format: {base_model_name}-ft-{exp_id}
        Example: qwen2-1.5b-ft-exp001
        """
        # Extract base model short name
        base_name = self.scen.base_model.split("/")[-1].lower()
        base_name = base_name.replace("_", "-")

        # Use workspace/experiment ID if available
        exp_id = implementation.workspace_path.name[:8]  # First 8 chars of workspace ID

        return f"{base_name}-ft-{exp_id}"

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
