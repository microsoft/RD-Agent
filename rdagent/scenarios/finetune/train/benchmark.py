"""
Benchmark Evaluation using OpenCompass

Evaluator that runs OpenCompass in Docker to evaluate fine-tuned models on standard benchmarks.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.env import BenchmarkDockerConf, BenchmarkDockerEnv


def _get_gpu_count() -> int:
    device_info_json = json.loads(get_runtime_environment_by_env(get_ft_env()))
    gpu_info = device_info_json.get("gpu", {})

    if "gpu_count" in gpu_info:
        return gpu_info["gpu_count"]

    if "gpus" in gpu_info:
        return len(gpu_info["gpus"])

    return 0


def _get_valid_tensor_parallel_size(num_gpus: int) -> int:
    """Adjust tensor_parallel_size to power of 2 (required by vLLM v1)"""
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
    """Create OpenCompass benchmark environment configuration"""
    conf = BenchmarkDockerConf()
    conf.running_timeout_period = FT_RD_SETTING.benchmark_timeout

    if FT_RD_SETTING.file_path:
        benchmark_cache_dir = Path(FT_RD_SETTING.file_path) / "benchmarks"
        benchmark_cache_dir.mkdir(parents=True, exist_ok=True)
        conf.extra_volumes = {str(benchmark_cache_dir.resolve()): {"bind": "/benchmarks", "mode": "rw"}}

    env = BenchmarkDockerEnv(conf=conf)
    env.prepare()

    num_gpus = _get_gpu_count()
    valid_tp_size = _get_valid_tensor_parallel_size(num_gpus)
    if valid_tp_size != num_gpus:
        logger.info(f"Adjusted tensor_parallel_size from {num_gpus} to {valid_tp_size}")

    env_vars = {
        "BENCHMARK_TASKS": ",".join(tasks),
        "ADAPTER_PATH": adapter_path,
        "BASE_MODEL": base_model,
        "OUTPUT_DIR": "/workspace/benchmark_results",
        "NUM_GPUS": str(valid_tp_size),
        "LIMIT": str(limit) if limit else "",
        "OC_JUDGE_MODEL": FT_RD_SETTING.judge_model,
        "OC_JUDGE_API_KEY": FT_RD_SETTING.judge_api_key or "",
        "OC_JUDGE_API_BASE": FT_RD_SETTING.judge_api_base or "",
        "MKL_THREADING_LAYER": "GNU",
    }

    return env, env_vars


class FTBenchmarkEvaluator(CoSTEEREvaluator):
    """Benchmark evaluator using OpenCompass in Docker"""

    def __init__(self, scen, tasks: Optional[List[str]] = None, limit: Optional[int] = None):
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
        """Run benchmark evaluation using OpenCompass"""
        workspace_path = implementation.workspace_path
        limit_str = f" (limit={self.limit})" if self.limit else ""
        logger.info(f"Benchmark eval: {self.tasks} on {self.scen.base_model}{limit_str}")

        env, env_vars = get_benchmark_env(
            tasks=self.tasks,
            adapter_path="/workspace/output",
            base_model=self.scen.base_model,
            limit=self.limit,
        )

        implementation.prepare()
        implementation.inject_files(**implementation.file_dict)

        result = env.run(
            entry="bash /app/opencompass_eval_entrypoint.sh",
            local_path=str(workspace_path),
            env=env_vars,
        )

        if result.exit_code != 0:
            return CoSTEERSingleFeedback(
                execution=f"Benchmark failed (exit_code={result.exit_code})",
                return_checking="OpenCompass evaluation error",
                code=result.stdout[-2000:] if result.stdout else "No output",
                final_decision=False,
            )

        results_path = workspace_path / "benchmark_results" / "results.json"
        scores = self._parse_results(results_path)
        report = self._format_report(scores)

        return CoSTEERSingleFeedback(
            execution=f"Benchmark completed: {len(scores)}/{len(self.tasks)} tasks",
            return_checking=report,
            code=f"Average Score: {self._calc_avg(scores):.2f}%",
            final_decision=True,
        )

    def _parse_results(self, results_path: Path) -> Dict[str, float]:
        """Parse OpenCompass results from JSON output"""
        if not results_path.exists():
            logger.warning(f"Results file not found: {results_path}")
            return {}

        try:
            with open(results_path) as f:
                data = json.load(f)

            scores = {}
            results = data.get("results", {})

            for task_name, task_results in results.items():
                for key, value in task_results.items():
                    if not key.endswith("_stderr") and isinstance(value, (int, float)):
                        scores[task_name] = value * 100 if 0 <= value <= 1 else value
                        break

            return scores
        except Exception as e:
            logger.error(f"Failed to parse results: {e}")
            return {}

    def _format_report(self, scores: Dict[str, float]) -> str:
        """Format results as text"""
        if not scores:
            return "No results available"

        lines = ["Benchmark Results:"]
        for task, score in scores.items():
            lines.append(f"  {task}: {score:.2f}%")
        return "\n".join(lines)

    def _calc_avg(self, scores: Dict[str, float]) -> float:
        """Calculate average score"""
        return sum(scores.values()) / len(scores) if scores else 0.0
