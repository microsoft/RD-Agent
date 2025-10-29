"""
Benchmark Evaluation using lm-evaluation-harness

Evaluator that runs lm-evaluation-harness in Docker to evaluate fine-tuned models on standard benchmarks.
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
from rdagent.utils.env import BenchmarkDockerConf, BenchmarkDockerEnv


# TODO: Do we need share runtime info in Scenario class?
def _get_gpu_count() -> int:
    """Get available GPU count, with fallback to 4."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 4  # Default fallback


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
        results_path = workspace_path / "benchmark_results" / "results.json"
        scores = self._parse_results(results_path)
        report = self._format_report(scores)

        return CoSTEERSingleFeedback(
            execution=f"Benchmark completed: {len(scores)}/{len(self.tasks)} tasks evaluated",
            return_checking=report,
            code=f"Average Score: {self._calc_avg(scores):.2f}",
            final_decision=True,
        )

    def _parse_results(self, results_path: Path) -> Dict[str, float]:
        """
        Parse benchmark results from lm-evaluation-harness JSON output.

        The results.json format from lm-evaluation-harness:
        {
          "results": {
            "gsm8k": {
              "exact_match,flexible-extract": 0.6875,
              "exact_match,strict-match": 0.296875,
              ...
            }
          }
        }
        """
        if not results_path.exists():
            logger.warning(f"Results file not found: {results_path}")
            return {}

        try:
            with open(results_path) as f:
                data = json.load(f)

            scores = {}
            results = data.get("results", {})

            for task_name, task_results in results.items():
                # Find the main metric (usually the first one without _stderr suffix)
                main_metric = None
                for key, value in task_results.items():
                    if not key.endswith("_stderr") and isinstance(value, (int, float)):
                        main_metric = value
                        break

                if main_metric is not None:
                    # Convert to percentage if it's a ratio
                    if 0 <= main_metric <= 1:
                        scores[task_name] = main_metric * 100
                    else:
                        scores[task_name] = main_metric

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
