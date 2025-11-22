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
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

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
    """
    Load benchmark dataset import paths from YAML configuration.

    Args:
        benchmark_name: Benchmark dataset name (e.g., "aime25", "gsm8k")

    Returns:
        List[str]: List of OpenCompass dataset module import paths

    Raises:
        FileNotFoundError: If benchmark_datasets.yaml not found
        ValueError: If benchmark_name not in configuration
    """
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
    """
    Load model inference configuration from YAML file.

    Args:
        base_model_name: HuggingFace model name (e.g., "Qwen/Qwen3-8B")

    Returns:
        dict: Merged configuration (model-specific overrides default)

    Raises:
        FileNotFoundError: If model_inference_configs.yaml not found
    """
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
    """
    Detect model type (full/LoRA) and extract base model name.

    Args:
        model_path: Path to fine-tuned model output directory

    Returns:
        tuple: (base_model_name, lora_adapter_path)
            - For LoRA: (base_model, model_path)
            - For full fine-tuning: (model_path, None)

    Raises:
        FileNotFoundError: If neither config.json nor adapter_config.json found
        ValueError: If base_model_name_or_path missing in config
    """
    model_dir = Path(model_path)

    # Check for LoRA adapter first (llama-factory format)
    adapter_config_file = model_dir / "adapter_config.json"
    if adapter_config_file.exists():
        with open(adapter_config_file, "r") as f:
            adapter_config = json.load(f)

        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError(f"Cannot find base_model_name_or_path in {adapter_config_file}")

        logger.info(f"Detected LoRA adapter at {model_path}, base model: {base_model}")
        return (base_model, str(model_dir.resolve()))

    # Check for full model config
    config_file = model_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"Neither config.json nor adapter_config.json found in {model_path}"
        )

    with open(config_file, "r") as f:
        config = json.load(f)

    # Extract base model name
    base_model = config.get("_name_or_path") or config.get("base_model_name_or_path")

    if not base_model:
        raise ValueError(f"Cannot find base model name in {config_file}")

    # Detect LoRA adapter files (alternative LoRA format)
    lora_files = [
        "adapter_model.bin",
        "adapter_model.safetensors",
    ]
    is_lora = any((model_dir / f).exists() for f in lora_files)

    if is_lora:
        logger.info(f"Detected LoRA adapter at {model_path}, base model: {base_model}")
        return (base_model, str(model_dir.resolve()))
    else:
        logger.info(f"Detected full fine-tuned model at {model_path}")
        return (str(model_dir.resolve()), None)


def run_benchmark(
    model_path: str,
    benchmark_name: str,
    limit: Optional[int] = None,
    num_runs: int = 1,
    pass_k: Optional[List[int]] = None,
    work_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run benchmark evaluation on a fine-tuned model.

    Args:
        model_path: Path to fine-tuned model (supports full/LoRA auto-detection)
        benchmark_name: Benchmark dataset name (e.g., "aime25", "gsm8k")
        limit: Optional dataset size limit for testing
        num_runs: Number of times to run each sample (default: 1)
        pass_k: Optional list of k values for pass@k evaluation (e.g., [1, 5, 10])
        work_dir: Optional work directory for results (default: temp dir)

    Returns:
        Dict[str, float]: Scores dictionary {task_name: score, ...}

    Raises:
        FileNotFoundError: Missing config files or model directory
        ValueError: Invalid benchmark name or configuration
        RuntimeError: Docker execution failed or results parsing error
    """
    # Load configurations我
    dataset_imports = get_benchmark_config(benchmark_name)
    base_model, lora_adapter = detect_model_type(model_path)
    inference_config = get_model_inference_config(base_model)

    # Prepare template variables (merge inference config from models.yaml)
    template_vars = {
        # Model configuration
        "model_abbr": f"ft-{benchmark_name}",
        "model_path": base_model,
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
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r") as f:
        template_content = f.read()

    template = Template(template_content)
    config_content = template.render(**template_vars)

    # Prepare Docker environment
    conf = BenchmarkDockerConf()
    conf.running_timeout_period = FT_RD_SETTING.benchmark_timeout

    # Prepare volume mounts
    extra_volumes = {}

    # Mount benchmark cache directory
    if FT_RD_SETTING.file_path:
        benchmark_cache_dir = Path(FT_RD_SETTING.file_path) / "benchmarks"
        try:
            benchmark_cache_dir.mkdir(parents=True, exist_ok=True)
            extra_volumes[str(benchmark_cache_dir.resolve())] = {"bind": "/benchmarks", "mode": "rw"}
            logger.info(f"Benchmark cache directory mounted: {benchmark_cache_dir}")
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot create benchmark cache directory {benchmark_cache_dir}: {e}. Skipping cache mount.")

    # Mount LoRA adapter directory if using LoRA
    if lora_adapter:
        lora_dir = Path(lora_adapter).resolve()
        if lora_dir.exists():
            extra_volumes[str(lora_dir)] = {"bind": str(lora_dir), "mode": "ro"}
            logger.info(f"LoRA adapter directory mounted: {lora_dir}")
        else:
            raise FileNotFoundError(f"LoRA adapter directory not found: {lora_dir}")

    conf.extra_volumes = extra_volumes
    env = BenchmarkDockerEnv(conf=conf)
    env.prepare()

    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config_file = temp_path / "config.py"

        with open(config_file, "w") as f:
            f.write(config_content)

        logger.info(f"Running benchmark '{benchmark_name}' on model: {model_path}")
        logger.info(f"Base model: {base_model}, LoRA: {lora_adapter is not None}")

        # Set environment variables
        env_vars = {
            "OC_JUDGE_MODEL": FT_RD_SETTING.judge_model,
            "OC_JUDGE_API_KEY": FT_RD_SETTING.judge_api_key or "",
            "OC_JUDGE_API_BASE": FT_RD_SETTING.judge_api_base or "",
        }

        # Run OpenCompass
        entry_cmd = f"opencompass /workspace/config.py --work-dir /workspace/benchmark_results"

        result = env.run(
            entry=entry_cmd,
            local_path=str(temp_path),
            env=env_vars,
        )

        # Check execution status
        if result.exit_code != 0:
            error_msg = result.stdout[-2000:] if result.stdout else "No output"
            raise RuntimeError(f"Benchmark execution failed (exit_code={result.exit_code})\n{error_msg}")

        # Parse and return results
        results_path = temp_path / "benchmark_results" / "results.json"
        scores = _parse_results(results_path)

        logger.info(f"Benchmark completed. Average score: {sum(scores.values()) / len(scores):.2f}%")
        return scores


def _parse_results(results_path: Path) -> Dict[str, float]:
    """
    Parse OpenCompass results from JSON output.

    Args:
        results_path: Path to results.json file

    Returns:
        Dict[str, float]: Scores dictionary

    Raises:
        FileNotFoundError: If results file not found
        ValueError: If results file is invalid or contains no scores
    """
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
    LORA_ADAPTER_PATH = "/home/v-qizhengli/workspace/FT_workspace/gitignore_folder/B200_FT_workspace/limo/train/b200_sweep_yamls/saves/qwen3-1.7b/lora_b200_lr1e-4_acc4/checkpoint-100"  # e.g., "/path/to/output/checkpoint-100"
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
        scores = run_benchmark(
            model_path=LORA_ADAPTER_PATH,
            benchmark_name=BENCHMARK,
        )

        print("\n✅ Evaluation completed!")
        for task, score in scores.items():
            print(f"  {task}: {score:.2f}%")

        avg_score = sum(scores.values()) / len(scores)
        print(f"\nAverage Score: {avg_score:.2f}%")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
