"""
Benchmark Evaluation using lm-evaluation-harness

Evaluator that runs lm-evaluation-harness in Docker to evaluate fine-tuned models on standard benchmarks.
"""

import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import BenchmarkDockerConf, BenchmarkDockerEnv


# TODO: Do we need share runtime info in Scenario class?
def _get_gpu_count() -> int:
    """Get available GPU count, with fallback to 4."""
    try:
        return torch.cuda.device_count()
    except Exception:
        pass
    return 1  # Default fallback


def _get_valid_tensor_parallel_size(num_gpus: int) -> int:
    """
    Adjust tensor_parallel_size to be a power of 2 (required by vLLM v1).

    Returns the largest power of 2 that is <= num_gpus.
    """
    if num_gpus <= 0:
        return 1
    power = 0
    while (1 << (power + 1)) <= num_gpus:
        power += 1
    return 1 << power


def get_benchmark_env(
    tasks: List[str],
    adapter_path: str,
    base_model: str,
    limit: Optional[int] = None,
) -> tuple:
    """
    Create lm-evaluation-harness benchmark environment with configuration.

    This function creates a specialized Docker environment for running lm-evaluation-harness benchmarks,
    separate from the training environment. The Dockerfile and eval_entrypoint.sh are located
    in rdagent/scenarios/finetune/docker/lm_eval/.

    Args:
        tasks: List of benchmark tasks (e.g., ['gsm8k', 'mmlu', 'hellaswag'])
        adapter_path: Absolute path to adapter files in workspace
        base_model: Base model identifier (e.g., 'Qwen/Qwen2.5-1.5B-Instruct')
        limit: Optional limit on number of samples to evaluate (for quick testing)

    Returns:
        Tuple of (env, env_vars) where:
        - env: BenchmarkDockerEnv configured for lm-evaluation-harness
        - env_vars: dict of environment variables to pass to Docker
    """

    # Create benchmark-specific Docker configuration
    conf = BenchmarkDockerConf()
    conf.running_timeout_period = FT_RD_SETTING.benchmark_timeout

    # Setup global benchmark cache directory (shared across all workspaces)
    if FT_RD_SETTING.file_path:
        from pathlib import Path

        benchmark_cache_dir = Path(FT_RD_SETTING.file_path) / "benchmarks"
        benchmark_cache_dir.mkdir(parents=True, exist_ok=True)
        conf.extra_volumes = {
            str(benchmark_cache_dir.resolve()): {
                "bind": "/benchmarks",
                "mode": "rw",
            }
        }

    # Create and prepare the benchmark environment
    env = BenchmarkDockerEnv(conf=conf)
    env.prepare()

    # Auto-detect GPU count and adjust to valid tensor_parallel_size
    num_gpus = _get_gpu_count()
    valid_tp_size = _get_valid_tensor_parallel_size(num_gpus)
    if valid_tp_size != num_gpus:
        logger.info(f"Adjusted tensor_parallel_size from {num_gpus} to {valid_tp_size} (vLLM requires power of 2)")

    # Environment variables to pass to Docker container
    # These will be read by eval_entrypoint.sh inside the container
    env_vars = {
        "BENCHMARK_TASKS": ",".join(tasks),
        "ADAPTER_PATH": adapter_path,
        "BASE_MODEL": base_model,
        "OUTPUT_DIR": "/workspace/benchmark_results",
        "BATCH_SIZE": "auto",
        "NUM_GPUS": str(valid_tp_size),  # Use valid power of 2
        # Set HF datasets cache to global benchmarks directory (mounted via extra_volumes)
        "HF_DATASETS_CACHE": "/benchmarks",
    }

    # Add limit if specified
    if limit is not None:
        env_vars["LIMIT"] = str(limit)

    return env, env_vars


class FTBenchmarkEvaluator(CoSTEEREvaluator):
    """
    Benchmark evaluator using lm-evaluation-harness in Docker.

    This evaluator runs standard LLM benchmarks on fine-tuned models.
    It should only run when training succeeds and adapter files are valid.
    """

    def __init__(self, scen, tasks: Optional[List[str]] = None, limit: Optional[int] = None):
        """
        Initialize benchmark evaluator.

        Args:
            scen: Scenario instance
            tasks: List of benchmark tasks to evaluate on
            limit: Optional limit on number of samples (None for full evaluation)
        """
        super().__init__(scen)
        self.tasks = tasks or FT_RD_SETTING.benchmark_datasets
        self.limit = limit

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: Optional[QueriedKnowledge] = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """
        Run benchmark evaluation using lm-evaluation-harness.

        Workflow:
        1. Launch lm-evaluation-harness Docker with environment variables
        2. Parse and report results
        """
        workspace_path = implementation.workspace_path
        output_path = workspace_path / "output"

        # Run benchmark in Docker
        logger.info(f"Starting benchmark evaluation on tasks: {self.tasks}")
        logger.info(f"Base model: {self.scen.base_model}")
        if self.limit:
            logger.info(f"Sample limit: {self.limit}")

        env, env_vars = get_benchmark_env(
            tasks=self.tasks,
            adapter_path="/workspace/output",  # Use Docker internal path
            base_model=self.scen.base_model,
            limit=self.limit,
        )

        # Prepare workspace and inject files
        implementation.prepare()
        implementation.inject_files(**implementation.file_dict)

        # Simple entrypoint script
        script_path = workspace_path / "run_benchmark.sh"
        script_path.write_text("#!/bin/bash\n" "cd /workspace\n" "bash /app/eval_entrypoint.sh\n")

        # FIXME: When task max_len exceeds model max_len, the evaluation will fail.
        # Run Docker with environment variables
        result = env.run(
            entry=f"bash {script_path.name}",
            local_path=str(workspace_path),
            env=env_vars,
        )

        if result.exit_code != 0:
            return CoSTEERSingleFeedback(
                execution=f"Benchmark execution failed (exit_code={result.exit_code})",
                return_checking="lm-evaluation-harness error",
                code=result.stdout[-2000:] if result.stdout else "No output",  # Last 2000 chars
                final_decision=False,
            )

        # Parse and format results
        benchmark_dir = workspace_path / "benchmark_results"

        if not benchmark_dir.exists():
            raise FileNotFoundError(f"benchmark_results directory does not exist at: {benchmark_dir}")

        # Find the most recent results file (handles both fixed and timestamped filenames)
        all_files = list(benchmark_dir.iterdir())
        logger.info(f"Files in benchmark_results directory: {[f.name for f in all_files]}")

        results_files = [f for f in all_files if f.name.startswith("results") and f.suffix == ".json"]
        if not results_files:
            raise FileNotFoundError(f"No results files found in {benchmark_dir}")

        # Use the most recent results file
        results_path = max(results_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using most recent results file: {results_path.name}")

        # Debug: Show file content preview
        logger.info(f"results file found! Size: {results_path.stat().st_size} bytes")
        try:
            with open(results_path, "r") as f:
                content = f.read()[:500]  # First 500 chars
            logger.info(f"results file content preview: {content}")
        except Exception as e:
            logger.error(f"Failed to read results file: {e}")

        scores = self._parse_results(results_path)
        report = self._format_report(scores)

        return CoSTEERSingleFeedback(
            execution=f"Benchmark completed: {len(scores)}/{len(self.tasks)} tasks evaluated",
            return_checking=report,
            code=f"Average Score: {self._calc_avg(scores):.2f}",
            final_decision=True,
        )

    def _parse_results(self, results_path: Path) -> Dict[str, float]:
        """Parse benchmark results from lm-evaluation-harness JSON output."""
        if not results_path.exists():
            logger.warning(f"Results file not found: {results_path}")
            return {}

        try:
            with open(results_path) as f:
                data = json.load(f)

            scores = {}
            results = data.get("results", {})

            for task_name, task_results in results.items():
                # Find the main metric (skip stderr entries)
                for key, value in task_results.items():
                    if not key.endswith("_stderr") and isinstance(value, (int, float)):
                        # Convert to percentage if it's a ratio
                        scores[task_name] = (value * 100) if 0 <= value <= 1 else value
                        logger.info(f"Final score for {task_name}: {scores[task_name]}")
                        break

            return scores

        except Exception as e:
            logger.error(f"Failed to parse results: {e}")
            return {}

    def _format_report(self, scores: Dict[str, float]) -> str:
        """Format results as text."""
        if not scores:
            return "No results available"

        lines = ["Benchmark Results:"]
        for task, score in scores.items():
            lines.append(f"  {task}: {score:.2f}%")
        return "\n".join(lines)

    def _calc_avg(self, scores: Dict[str, float]) -> float:
        """Calculate average score."""
        return sum(scores.values()) / len(scores) if scores else 0.0
