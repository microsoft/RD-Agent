"""
Benchmark Evaluation using OpenCompass

Evaluator that runs OpenCompass in Docker to evaluate fine-tuned models on standard benchmarks.

Configure benchmark behavior via editting .env to cover default settings in conf.py:
```
FT_BENCHMARK_DATASETS='["aime25", "gsm8k"]'
FT_BENCHMARK_NUM_RUNS=4
FT_JUDGE_MODEL="gpt-4"
FT_JUDGE_API_KEY="sk-xxx"
FT_JUDGE_API_BASE="https://api.openai.com/v1"
```
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv
from jinja2 import Template

# Load .env file before importing settings
load_dotenv(find_dotenv())

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


def get_benchmark_config(benchmark_name: str) -> List[str]:
    """Load benchmark dataset import paths from YAML configuration."""
    config_path = Path(__file__).parent / "benchmark_configs" / "datasets.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if benchmark_name not in config_data:
        available = ", ".join(config_data.keys())
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {available}")

    imports = config_data[benchmark_name]
    return [imports] if isinstance(imports, str) else imports


def get_model_inference_config(base_model_name: str) -> dict:
    """Load model inference configuration from YAML file."""
    config_path = Path(__file__).parent / "benchmark_configs" / "models.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Model inference config not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    default_config = config_data.get("default", {})
    model_configs = config_data.get("models", {})
    model_specific = model_configs.get(base_model_name, {})

    final_config = {**default_config, **model_specific}

    # Handle auto tensor_parallel_size
    if final_config.get("tensor_parallel_size") == "auto":
        num_gpus = _get_gpu_count()
        final_config["tensor_parallel_size"] = _get_valid_tensor_parallel_size(num_gpus)

    return final_config


def detect_model_type(model_path: str) -> tuple[str, Optional[str]]:
    """Detect if the model is fine-tuned with LoRA or full fine-tuning."""
    model_dir = Path(model_path)

    # Check for LoRA adapter first (llama-factory format)
    adapter_config_file = model_dir / "adapter_config.json"
    if adapter_config_file.exists():
        with open(adapter_config_file, "r") as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get("base_model_name_or_path")

        logger.info(f"Detected LoRA adapter at {model_path}, base model: {base_model}")
        base_model_dir = FT_RD_SETTING.file_path / "models" / base_model
        return (base_model_dir, str(model_dir.resolve()))

    # Not LoRA, must be full fine-tuning
    logger.info(f"Detected full fine-tuned model at {model_path}")
    return (model_path, None)


def _run_benchmark_in_workspace(
    workspace_path: Path,
    config_content: str,
    benchmark_name: str,
    model_path: str,
    base_model: str,
    lora_adapter: Optional[str],
    work_dir: str,
    env: BenchmarkDockerEnv,
) -> Dict[str, float]:
    """Execute benchmark evaluation in workspace."""
    # Write config file
    config_file = workspace_path / "config.py"
    config_file.write_text(config_content)

    # Convert work_dir to container path
    rel_path = Path(work_dir).relative_to(workspace_path)
    docker_work_dir = f"/workspace/{rel_path}"

    # Logging
    logger.info(f"Running benchmark '{benchmark_name}' on model: {model_path}")
    logger.info(f"Base model: {base_model}, LoRA?: {lora_adapter is not None}")
    logger.info(f"Workspace: {workspace_path}")
    logger.info(f"Docker work_dir: {docker_work_dir}")

    # Environment variables
    env_vars = {
        "OC_JUDGE_MODEL": FT_RD_SETTING.judge_model,
        "OC_JUDGE_API_KEY": FT_RD_SETTING.judge_api_key,
        "OC_JUDGE_API_BASE": FT_RD_SETTING.judge_api_base,
    }

    # Run OpenCompass
    entry_cmd = f"opencompass /workspace/config.py --work-dir {docker_work_dir}"

    result = env.run(
        entry=entry_cmd,
        local_path=str(workspace_path),
        env=env_vars,
    )

    # Scan for timestamped directories after execution
    results_base = workspace_path / rel_path
    timestamped_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()], reverse=True)
    # OpenCompass stores results in results/<model_name>/<dataset>.json
    results_subdir = timestamped_dirs[0] / "summary"

    csv_files = sorted([f for f in results_subdir.rglob("*.csv")], reverse=True)
    results_csv_path = csv_files[0]
    logger.info(f"Detailed results CSV: {results_csv_path.relative_to(results_base)}")

    # Read and return CSV content
    df = pd.read_csv(results_csv_path)
    return df.to_dict('records')


def run_benchmark(
    model_path: str,
    benchmark_name: str,
    work_dir: str,
    limit: Optional[int] = None,
    num_runs: int = 1,
    pass_k: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Run benchmark evaluation on a fine-tuned model.

    Args:
        model_path: Path to fine-tuned model (supports full/LoRA auto-detection)
        benchmark_name: Benchmark dataset name (e.g., "aime25", "gsm8k")
        work_dir: Work directory for results (within FBWorkspace)
        limit: Optional dataset size limit for testing
        num_runs: Number of times to run each sample (default: 1)
        pass_k: Optional list of k values for pass@k evaluation (e.g., [1, 5, 10])

    Returns:
        Dict[str, float]: Scores dictionary {task_name: score, ...}
    """
    
    dataset_imports = get_benchmark_config(benchmark_name)  # Load dataset configuration
    model, lora_adapter = detect_model_type(model_path)    # Detect model type (LoRA or full fine-tuning)
    inference_config = get_model_inference_config(model)   # Load model inference configuration

    # Prepare template variables (merge inference config from models.yaml)
    template_vars = {
        # Model configuration
        "model_abbr": f"ft-{benchmark_name}",
        "model_path": model,
        "is_lora": lora_adapter is not None,
        "lora_path": lora_adapter or "",
        # Dataset configuration
        "dataset_imports": dataset_imports,
        "limit": limit or "",
        "num_runs": num_runs,
        "pass_k": pass_k,
        "work_dir": work_dir or "/workspace/benchmark_results",
        # Merge all inference parameters from models.yaml (default + model-specific)
        **inference_config,
    }

    # Render Jinja2 template
    template_path = Path(__file__).parent / "benchmark_configs" / "opencompass_template.py.j2"
    with open(template_path, "r") as f:
        template_content = f.read()

    template = Template(template_content)
    config_content = template.render(**template_vars)

    # Prepare Docker environment
    conf = BenchmarkDockerConf()
    conf.running_timeout_period = FT_RD_SETTING.benchmark_timeout

    # Setup volume mounts
    extra_volumes = {}

    # Setup benchmark cache volume
    if FT_RD_SETTING.file_path:
        cache_dir = Path(FT_RD_SETTING.file_path) / "benchmarks"
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Benchmark cache mounted: {cache_dir}")
        extra_volumes[str(cache_dir.resolve())] = {"bind": "/benchmarks", "mode": "rw"}

    # Setup LoRA volume
    if lora_adapter:
        lora_dir = Path(lora_adapter).resolve()
        extra_volumes[str(lora_dir)] = {"bind": "/lora_adapter", "mode": "ro"}
        base_model_dir = FT_RD_SETTING.file_path / "base_models" / model
        extra_volumes[base_model_dir] = {"bind": "/model", "mode": "ro"}
    else :
        extra_volumes[model_path] = {"bind": "/model", "mode": "ro"}

    conf.extra_volumes = extra_volumes
    env = BenchmarkDockerEnv(conf=conf)
    env.prepare()

    # Execute benchmark in FBWorkspace
    workspace_path = Path(work_dir).parent
    workspace_path.mkdir(parents=True, exist_ok=True)
    return _run_benchmark_in_workspace(
        workspace_path, config_content, benchmark_name,
        model_path, model, lora_adapter, work_dir, env
    )


def _parse_results(results_path: Path) -> Dict[str, float]:
    """Parse OpenCompass results from JSON output."""
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    try:
        with open(results_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in results file: {e}")

    results = data.get("results", {})
    if not results:
        raise ValueError("No results found in output")

    scores = {}
    for task_name, task_results in results.items():
        for key, value in task_results.items():
            if not key.endswith("_stderr") and isinstance(value, (int, float)):
                scores[task_name] = value * 100 if 0 <= value <= 1 else value
                break

    if not scores:
        raise ValueError("Failed to extract scores from results")

    return scores


class FTBenchmarkEvaluator(CoSTEEREvaluator):
    """
    Benchmark evaluator using OpenCompass in Docker (Backward Compatible)

    Wraps the simplified run_benchmark function to maintain compatibility
    with CoSTEER framework. Raises exceptions directly on any errors.
    """

    def __init__(
        self,
        scen,
        tasks: Optional[List[str]] = None,
        limit: Optional[int] = None,
        num_runs: Optional[int] = None,
        pass_k: Optional[List[int]] = None,
    ):
        super().__init__(scen)
        self.tasks = tasks or FT_RD_SETTING.benchmark_datasets
        self.limit = limit
        self.num_runs = num_runs if num_runs is not None else FT_RD_SETTING.benchmark_num_runs
        self.pass_k = pass_k if pass_k is not None else FT_RD_SETTING.benchmark_pass_k

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: Optional[QueriedKnowledge] = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """
        Run benchmark evaluation using the simplified run_benchmark function.

        Evaluates all benchmarks in self.tasks and aggregates results.
        Raises exceptions directly if any error occurs during evaluation.
        """
        workspace_path = implementation.workspace_path
        limit_str = f" (limit={self.limit})" if self.limit else ""
        num_runs_str = f" (num_runs={self.num_runs})" if self.num_runs > 1 else ""
        pass_k_str = f" (pass_k={self.pass_k})" if self.pass_k else ""
        model_path = str((workspace_path / "output").resolve())

        logger.info(f"Benchmark eval: {self.tasks} on model at {model_path}{limit_str}{num_runs_str}{pass_k_str}")

        if not self.tasks:
            raise ValueError("No benchmark tasks specified for evaluation")

        # Run all benchmarks and collect results
        all_scores = {}
        for benchmark_name in self.tasks:
            logger.info(f"Running benchmark: {benchmark_name}")
            scores = run_benchmark(
                model_path=model_path,
                benchmark_name=benchmark_name,
                limit=self.limit,
                num_runs=self.num_runs,
                pass_k=self.pass_k,
                work_dir=str((workspace_path / "benchmark_results" / benchmark_name).resolve()),
            )
            # Prefix task names with benchmark name to avoid conflicts
            for task_name, score in scores.items():
                all_scores[f"{benchmark_name}/{task_name}"] = score

        # Calculate overall average
        if all_scores:
            overall_average = sum(all_scores.values()) / len(all_scores)
        else:
            overall_average = 0.0

        details = self._format_report(all_scores)

        return CoSTEERSingleFeedback(
            execution=f"Benchmark completed: {', '.join(self.tasks)}",
            return_checking=details,
            code=f"Overall Average Score: {overall_average:.2f}%",
            final_decision=True,
        )

    def _format_report(self, scores: Dict[str, float]) -> str:
        """Format results as text"""
        lines = ["Benchmark Results:"]
        for task, score in scores.items():
            lines.append(f"  {task}: {score:.2f}%")
        return "\n".join(lines)


if __name__ == "__main__":
    """Test benchmark evaluation on Qwen3-1.7B with LoRA adapter."""
    # Configuration - Fill in your LoRA adapter path
    LORA_ADAPTER_PATH = "/home/v-qizhengli/workspace/FT_workspace/gitignore_folder/B200/B200_FT_workspace/limo/train/b200_sweep_yamls/saves/qwen3-1.7b/lora_b200_lr1e-4_acc4/checkpoint-100"  # e.g., "/path/to/output/checkpoint-100"
    BENCHMARK = "aime25"

    print("=" * 80)
    print("Benchmark Evaluation Test")
    print("=" * 80)
    print(f"\n📋 Environment: FT_JUDGE_API_KEY={'✅ Set' if FT_RD_SETTING.judge_api_key else '❌ Not Set'}")
    print(f"   Judge API Base: {FT_RD_SETTING.judge_api_base or '❌ Not Set'}")

    if LORA_ADAPTER_PATH is None:
        print("\n⚠️  Please set LORA_ADAPTER_PATH to your LoRA checkpoint directory")
        print('   Example: LORA_ADAPTER_PATH = "/workspace/output"')
        exit(1)

    print(f"\nModel: {LORA_ADAPTER_PATH}")
    print(f"Benchmark: {BENCHMARK}")
    print("-" * 80)

    try:
        # Create FBWorkspace for test (auto-generates UUID workspace)
        test_task = Task(name=f"benchmark_test_{BENCHMARK}")
        test_workspace = FBWorkspace(target_task=test_task)
        test_workspace.prepare()

        print(f"\n📁 Workspace: {test_workspace.workspace_path}")

        # Set work_dir to workspace subdirectory
        work_dir = str((test_workspace.workspace_path / "benchmark_results").resolve())

        scores = run_benchmark(
            model_path=LORA_ADAPTER_PATH,
            benchmark_name=BENCHMARK,
            work_dir=work_dir,
        )

        print("\n✅ Evaluation completed!")
        for task, score in scores.items():
            print(f"  {task}: {score:.2f}%")

        avg_score = sum(scores.values()) / len(scores)
        print(f"\nAverage Score: {avg_score:.2f}%")
        print(f"\n📂 Results saved to: {work_dir}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
