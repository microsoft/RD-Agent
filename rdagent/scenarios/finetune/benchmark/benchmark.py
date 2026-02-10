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
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import (
    FT_MODEL_PATH,
    get_benchmark_env,
    get_ft_env,
    get_workspace_prefix,
    is_docker_env,
)
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.scenarios.finetune.benchmark.data.adaptor import (
    BENCHMARK_CONFIG_DICT,
    BenchmarkConfig,
)
from rdagent.scenarios.finetune.benchmark.data.default import extract_error_samples
from rdagent.scenarios.finetune.benchmark.merge.merge import (
    check_if_merging_needed,
    merge_model,
)
from rdagent.utils.agent.tpl import T


def get_model_inference_config(base_model_name: str, gpu_count: int) -> dict:
    """
    Load model inference configuration from YAML file.

    Args:
        base_model_name: HuggingFace model name (e.g., "Qwen/Qwen3-8B")
        gpu_count: GPU count for tensor_parallel_size (from scenario.device_info)

    Returns:
        dict: Merged configuration (model-specific overrides default)
              Uses exact match first, then longest prefix match, finally default only.
    """
    from rdagent.components.benchmark import BENCHMARK_CONFIGS_DIR
    config_data = yaml.safe_load(open(BENCHMARK_CONFIGS_DIR / "models.yaml", "r"))

    default_config = config_data.get("default", {})
    models_config = config_data.get("models", {})

    # 1. Exact match
    if base_model_name in models_config:
        model_specific = models_config[base_model_name]
    else:
        # 2. Prefix match - find longest matching prefix
        model_specific = {}
        best_match_len = 5
        for configured_model in models_config:
            if base_model_name.startswith(configured_model) and len(configured_model) > best_match_len:
                model_specific = models_config[configured_model]
                best_match_len = len(configured_model)

    final_config = {**default_config, **model_specific}

    # Handle auto tensor_parallel_size
    if final_config.get("tensor_parallel_size") == "auto":
        if gpu_count <= 0:
            final_config["tensor_parallel_size"] = 1
        else:
            # Round down to nearest power of 2
            power = 0
            while (1 << (power + 1)) <= gpu_count:
                power += 1
            final_config["tensor_parallel_size"] = 1 << power

    return final_config


def detect_model_type(model_path: str) -> bool:
    """
    Detect whether the given model path corresponds to a LoRA adapter.

    Returns:
        True if LoRA adapter, False otherwise.
    """
    model_dir = Path(model_path)

    # LoRA (llama-factory style)
    if (model_dir / "adapter_config.json").exists():
        return True

    # Alternate LoRA file indicators
    for fname in ("adapter_model.bin", "adapter_model.safetensors"):
        if (model_dir / fname).exists():
            return True

    return False


def run_benchmark(
    workspace_path: str,
    model_path: str,
    model_name: str,
    benchmark_name: str,
    gpu_count: int,
    test_range: Optional[str] = "[:100]",
    num_runs: int = 1,
    pass_k: Optional[List[int]] = None,
    max_error_samples: int = 10,
    result_subdir: str = "",
) -> Dict[str, Any]:
    """
    Run benchmark evaluation on a fine-tuned model.

    Args:
        workspace_path: Path to workspace directory
        model_path: Path to fine-tuned model (supports full/LoRA auto-detection)
        model_name: HuggingFace model name
        benchmark_name: Benchmark dataset name (e.g., "aime25", "gsm8k")
        gpu_count: GPU count for tensor_parallel_size (from scenario.device_info)
        test_range: Python slice string for dataset sampling (e.g., "[:100]", "[-100:]").
                    Negative indexing allows automatic adaptation to varying subset sizes.
        num_runs: Number of times to run each sample (default: 1)
        pass_k: Optional list of k values for pass@k evaluation (e.g., [1, 5, 10])
        max_error_samples: Maximum number of error samples to extract for feedback
        result_subdir: Subdirectory for results (e.g., "validation", "test")

    Returns:
        Dict containing:
        - accuracy_summary: Dict mapping dataset -> {metric: value}, grouped by dataset
        - error_samples: List of error samples for feedback analysis
    """
    # Load configurations
    benchmark_cfg: BenchmarkConfig = BENCHMARK_CONFIG_DICT[benchmark_name]
    dataset_imports = benchmark_cfg.dataset

    # Auto download dependent data if configured on this benchmark
    if benchmark_cfg.download is not None:
        benchmark_cfg.download()

    model_is_lora = detect_model_type(model_path)
    inference_config = get_model_inference_config(model_name, gpu_count)
    workspace_path = Path(workspace_path)

    # Get environment first to determine path prefix
    env = get_benchmark_env()
    ws_prefix = get_workspace_prefix(env)
    is_docker = is_docker_env(env)

    # Determine model paths based on environment type
    model_rel_path = Path(model_path).relative_to(workspace_path)
    adapter_path_in_env = Path(ws_prefix) / model_rel_path

    if model_is_lora:
        if is_docker:
            # Docker: use /assets/models mount
            model_path_in_env = Path(FT_MODEL_PATH) / model_name
        else:
            # Conda: use actual file path
            model_path_in_env = Path(FT_RD_SETTING.file_path) / "models" / model_name
        lora_path_in_env = adapter_path_in_env

        # Check if we need to merge the model (e.g. vLLM doesn't support LoRA with modules_to_save)
        if check_if_merging_needed(model_path):
            merged_model_dir_inside_env = Path(ws_prefix) / "merged_model"

            # Create a temporary environment for merging (use FT env as it has peft/transformers)
            merge_env = get_ft_env()

            merge_model(
                env=merge_env,
                workspace_path=workspace_path,
                base_model_path=str(model_path_in_env),
                adapter_path=str(lora_path_in_env),
                output_path=str(merged_model_dir_inside_env),
            )

            # Switch to using the merged model
            model_path_in_env = merged_model_dir_inside_env
            model_is_lora = False
            lora_path_in_env = ""
            adapter_path_in_env = merged_model_dir_inside_env
    else:
        model_path_in_env = adapter_path_in_env
        lora_path_in_env = ""

    # Prepare template variables (merge inference config from models.yaml)
    template_vars = {
        # Model configuration
        "model_abbr": f"ft-{benchmark_name}",
        "model_path": model_path_in_env,
        "is_lora": model_is_lora,
        "lora_path": lora_path_in_env,
        # Dataset configuration
        "dataset_imports": [dataset_imports],
        "test_range": test_range,
        "num_runs": num_runs,
        "pass_k": pass_k,
        "work_dir": adapter_path_in_env,
        # Merge all inference parameters from models.yaml (default + model-specific)
        **inference_config,
    }

    # Override use_cot_postprocessor based on force_think_token setting
    # When force_think_token=false, we don't need the CoT postprocessor to extract answers
    if not FT_RD_SETTING.force_think_token:
        template_vars["use_cot_postprocessor"] = False

    # Render Jinja2 template
    config_content = T("rdagent.components.benchmark.configs.opencompass_template:template").r(**template_vars)

    # Note: env was already created above via get_benchmark_env()

    (workspace_path / "config.py").write_text(config_content)
    # Use result_subdir for validation/test separation
    if result_subdir:
        benchmark_work_dir = f"{ws_prefix}/benchmark_results/{result_subdir}"
    else:
        benchmark_work_dir = f"{ws_prefix}/benchmark_results"

    # Logging
    logger.info(f"Running benchmark '{benchmark_name}' on model: {model_path}")
    logger.info(f"Base model: {model_name}, LoRA?: {model_is_lora}")
    logger.info(f"Workspace: {workspace_path}")
    logger.info(f"Benchmark work_dir: {benchmark_work_dir}")
    if test_range:
        logger.info(f"Dataset range: {test_range}")

    # Environment variables
    env_vars = {
        "OC_JUDGE_MODEL": FT_RD_SETTING.judge_model or LLM_SETTINGS.chat_model,
        "OC_JUDGE_API_KEY": FT_RD_SETTING.judge_api_key or LLM_SETTINGS.openai_api_key,
        "OC_JUDGE_API_BASE": FT_RD_SETTING.judge_api_base or LLM_SETTINGS.openai_api_base,
        "OC_JUDGE_RETRY": str(FT_RD_SETTING.judge_retry),
    }

    # Check if results already exist (skip re-running if cached)
    results_base = workspace_path / "benchmark_results"
    if result_subdir:
        results_base = results_base / result_subdir
    timestamped_dirs = sorted([d for d in results_base.glob("202*_*") if d.is_dir()], reverse=True)

    if timestamped_dirs:
        logger.info(f"Found existing results in {timestamped_dirs[0].name}, skipping benchmark execution")
    else:
        # Run OpenCompass
        entry_cmd = f"opencompass {ws_prefix}/config.py --work-dir {benchmark_work_dir}"

        result = env.run(
            entry=entry_cmd,
            local_path=str(workspace_path),
            env=env_vars,
        )

        # Log execution immediately (for UI display)
        tag_prefix = "docker_run" if is_docker else "conda_run"
        logger.log_object(
            {
                "exit_code": result.exit_code,
                "stdout": (result.stdout or ""),
                "benchmark_name": benchmark_name,
                "model_path": str(model_path),
                "workspace_path": str(workspace_path),
            },
            tag=f"{tag_prefix}.Benchmark",
        )

        # Check execution status
        if result.exit_code != 0:
            error_msg = result.stdout[-2000:] if result.stdout else "No output"
            raise RuntimeError(f"Benchmark execution failed (exit_code={result.exit_code})\n{error_msg}")

        # Re-scan for timestamped directories after execution
        timestamped_dirs = sorted([d for d in results_base.glob("202*_*") if d.is_dir()], reverse=True)

    # OpenCompass stores results in results/<model_name>/<dataset>.json
    results_subdir = timestamped_dirs[0] / "summary"

    results_csv_path = sorted([f for f in results_subdir.rglob("*.csv")], reverse=True)[0]
    logger.info(f"Detailed results CSV: {results_csv_path.relative_to(results_base)}")

    # Read CSV content for accuracy summary (grouped by dataset)
    df = pd.read_csv(results_csv_path)
    # Get score column (the model name column, e.g., 'api-chemcotbench')
    score_col = [c for c in df.columns if c not in ["dataset", "version", "metric", "mode"]][0]
    # Pivot to group by dataset, with metrics as columns (use pivot_table to handle duplicates)
    pivoted = df.pivot_table(index="dataset", columns="metric", values=score_col, aggfunc="first").to_dict("index")
    # Filter out NaN values (different datasets have different metrics)
    accuracy_summary = {ds: {k: v for k, v in metrics.items() if pd.notna(v)} for ds, metrics in pivoted.items()}

    # Extract error samples for feedback
    error_samples = extract_error_samples(
        timestamped_dirs[0],
        max_samples=max_error_samples,
    )

    # Log benchmark result for UI display
    # Use result_subdir to distinguish validation vs test in tag
    log_tag = f"benchmark_result.{result_subdir}" if result_subdir else "benchmark_result"
    logger.log_object(
        {
            "accuracy_summary": accuracy_summary,
            "error_samples": error_samples,
            "benchmark_name": benchmark_name,
            "split": result_subdir or "default",  # validation, test, or default
        },
        tag=log_tag,
    )

    return {
        "accuracy_summary": accuracy_summary,
        "error_samples": error_samples,
    }


def get_benchmark_ranges() -> tuple[str, str]:
    """Get validation and test range strings for benchmark evaluation.

    Uses dynamic expressions that adapt to any dataset size:
    - For small datasets (<200): splits 50/50 to avoid overlap
    - For large datasets (>=200): takes 100 samples each

    The expressions use OpenCompass's eval mechanism with index_list variable.

    Returns:
        Tuple of (validation_range, test_range) - guaranteed non-overlapping:
        - validation: first min(100, 50%) samples
        - test: last min(100, 50%) samples
    """
    return "[:min(100, len(index_list)//2)]", "[-min(100, len(index_list)//2):]"


if __name__ == "__main__":
    """Test benchmark evaluation on Qwen3-1.7B with LoRA adapter."""
    # Configuration - Fill in your LoRA adapter path and model name
    LORA_ADAPTER_PATH = "/home/v-qizhengli/workspace/FT_workspace/gitignore_folder/B200/B200_FT_workspace/limo/train/b200_sweep_yamls/saves/qwen3-1.7b/lora_b200_lr1e-4_acc4/checkpoint-100"
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    BENCHMARK = "aime25"
    GPU_COUNT = 1

    print("=" * 80)
    print("Benchmark Evaluation Test")
    print("=" * 80)
    print(f"\nEnvironment: FT_JUDGE_API_KEY={'Set' if FT_RD_SETTING.judge_api_key else 'Not Set'}")
    print(f"Judge API Base: {FT_RD_SETTING.judge_api_base or 'Not Set'}")

    if not Path(LORA_ADAPTER_PATH).exists():
        print(f"\nPlease set LORA_ADAPTER_PATH to a valid checkpoint directory")
        print(f"Current path does not exist: {LORA_ADAPTER_PATH}")
        exit(1)

    print(f"\nModel: {MODEL_NAME}")
    print(f"Adapter: {LORA_ADAPTER_PATH}")
    print(f"Benchmark: {BENCHMARK}")
    print("-" * 80)

    try:
        # Create FBWorkspace for test (auto-generates UUID workspace)
        test_task = Task(name=f"benchmark_test_{BENCHMARK}")
        test_workspace = FBWorkspace(target_task=test_task)
        test_workspace.prepare()

        print(f"\nWorkspace: {test_workspace.workspace_path}")

        result = run_benchmark(
            workspace_path=str(test_workspace.workspace_path),
            model_path=LORA_ADAPTER_PATH,
            model_name=MODEL_NAME,
            benchmark_name=BENCHMARK,
            gpu_count=GPU_COUNT,
        )

        print("\nEvaluation completed!")
        print(f"Accuracy Summary: {result['accuracy_summary']}")
        print(f"Error Samples: {len(result['error_samples'])} samples")
        print(f"\nResults saved to: {test_workspace.workspace_path / 'benchmark_results'}")

    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback

        traceback.print_exc()
