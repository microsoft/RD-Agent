"""
Standalone test script for testing extract_error_samples.

Usage:
    python test_benchmark.py

Uses rdagent's Docker environment with cache enabled.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# Set FT_file_path BEFORE importing rdagent modules (so Docker mounts correct path)
_project_root = Path(__file__).resolve().parents[2]
os.environ["FT_file_path"] = str(_project_root / "git_ignore_folder" / "finetune_files")

import pandas as pd

from rdagent.components.coder.finetune.conf import get_benchmark_env
from rdagent.scenarios.finetune.benchmark.data.adaptor import BENCHMARK_CONFIG_DICT
from rdagent.scenarios.finetune.benchmark.data.default import extract_error_samples
from rdagent.utils.agent.tpl import T


def run_benchmark_simple(
    workspace_path: str,
    model_path_in_docker: str,
    benchmark_name: str,
    gpu_count: int = 4,
    limit: int = 3,
    offset: int = 0,
    max_error_samples: int = 5,
    result_subdir: str = "",
):
    """
    Simplified benchmark runner using rdagent Docker env.

    Args:
        workspace_path: Local workspace path
        model_path_in_docker: Model path inside Docker (e.g., /finetune/models/Qwen/Qwen2.5-1.5B)
        benchmark_name: Benchmark name
        gpu_count: GPU count
        limit: Dataset limit
        offset: Starting offset for dataset sampling (default: 0)
        max_error_samples: Max error samples to extract
        result_subdir: Subdirectory for results (e.g., "validation", "test")
    """
    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)

    cfg = BENCHMARK_CONFIG_DICT[benchmark_name]

    # Auto download dependent data if configured
    if cfg.download is not None:
        cfg.download()

    # Calculate tensor_parallel_size (round down to power of 2)
    tp_size = 1
    power = 0
    while (1 << (power + 1)) <= gpu_count:
        power += 1
    tp_size = 1 << power

    # Generate config.py (paths are Docker paths)
    config_content = T("rdagent.scenarios.finetune.benchmark.configs.opencompass_template:template").r(
        model_abbr=f"test-{benchmark_name}",
        model_path=model_path_in_docker,
        is_lora=False,
        lora_path="",
        dataset_imports=[cfg.dataset],
        limit=limit,
        offset=offset,
        num_runs=1,
        pass_k=None,
        work_dir="/workspace",  # Docker workspace path
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        max_seq_len=32768,
        max_out_len=8192,
        batch_size=16,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
        enable_thinking=False,
    )

    config_file = workspace / "config.py"
    config_file.write_text(config_content)

    # Get Docker env with cache enabled
    env = get_benchmark_env()
    env.conf.enable_cache = True

    # Environment variables for LLM judge (required for cascade eval benchmarks like AIME25)
    env_vars = {
        "OC_JUDGE_MODEL": "gpt-5.1",
        "OC_JUDGE_API_KEY": "sk-1234",
        "OC_JUDGE_API_BASE": "http://localhost:3000",
        "OC_JUDGE_RETRY": "3",
    }

    # Run opencompass in Docker
    if result_subdir:
        benchmark_work_dir = f"/workspace/benchmark_results/{result_subdir}"
    else:
        benchmark_work_dir = "/workspace/benchmark_results"
    cmd = f"opencompass /workspace/config.py --work-dir {benchmark_work_dir}"
    print(f"Running in Docker: {cmd}")
    if offset:
        print(f"Dataset range: [{offset}:{offset + limit}]")

    result = env.run(
        entry=cmd,
        local_path=str(workspace),
        env=env_vars,
    )

    print(f"Exit code: {result.exit_code}")
    if result.exit_code != 0:
        print(f"Error: {result.stdout[-2000:] if result.stdout else 'No output'}")
        raise RuntimeError(f"Benchmark failed with exit code {result.exit_code}")

    # Extract results from local workspace
    work_dir = workspace / "benchmark_results"
    if result_subdir:
        work_dir = work_dir / result_subdir
    timestamped_dirs = sorted(work_dir.glob("202*_*"), reverse=True)
    if not timestamped_dirs:
        raise RuntimeError(f"No results found in {work_dir}")

    result_dir = timestamped_dirs[0]
    csv_files = sorted(result_dir.rglob("summary/*.csv"), reverse=True)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {result_dir}")

    # Parse benchmark results from CSV, grouped by dataset
    df = pd.read_csv(csv_files[0])
    # Get score column (the model name column, e.g., 'test-chemcotbench')
    score_col = [c for c in df.columns if c not in ["dataset", "version", "metric", "mode"]][0]
    # Pivot to group by dataset, with metrics as columns (use pivot_table to handle duplicates)
    pivoted = df.pivot_table(index="dataset", columns="metric", values=score_col, aggfunc="first").to_dict("index")
    # Filter out NaN values (different datasets have different metrics)
    benchmark_results = {ds: {k: v for k, v in metrics.items() if pd.notna(v)} for ds, metrics in pivoted.items()}

    # Extract error samples
    errors = extract_error_samples(
        result_dir,
        max_samples=max_error_samples,
    )

    return {"benchmark_results": benchmark_results, "error_samples": errors}


if __name__ == "__main__":
    # Change to project root (required for template resolution)
    os.chdir(_project_root)

    # Configuration
    MODEL = "Qwen/Qwen3-8B"
    LIMIT = 3
    GPU_COUNT = 4

    # Docker model path (models are mounted at /finetune/models)
    model_path_in_docker = f"/finetune/models/{MODEL}"

    # Create test directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_base = _project_root / "git_ignore_folder" / "test" / timestamp

    print("=" * 60)
    print(f"BENCHMARK TEST: {MODEL} (limit={LIMIT})")
    print(f"Docker model path: {model_path_in_docker}")
    print(f"Output: {test_base}")
    print("=" * 60)

    results_summary = {}

    # Hardcoded benchmark list - comment/uncomment to select benchmarks to test
    BENCHMARKS_TO_TEST = [
        # Math Reasoning
        # "aime24",
        # "aime25",
        # "math",
        # General Knowledge
        # "mmlu",
        # Code Generation
        # "humaneval",
        # "mbpp",
        # PANORAMA - Patent Analysis (zero-shot)
        # "panorama",
        # "panorama_par4pc",
        # "panorama_pi4pc",
        # "panorama_noc4pc",
        # PANORAMA - Patent Analysis (CoT)
        # "panorama_par4pc_cot",
        # "panorama_pi4pc_cot",
        # "panorama_noc4pc_cot",
        # ChemCoTBench - Chemistry Reasoning
        # "chemcotbench",
        "chemcotbench_mol_und",
        "chemcotbench_mol_edit",
        "chemcotbench_mol_opt",
        "chemcotbench_reaction",
        # TableBench - Table QA
        "tablebench_data_analysis",
        "tablebench_fact_checking",
        "tablebench_numerical_reasoning",
        "tablebench_visualization",
        # "tablebench_gen",
        # Finance
        # "FinanceIQ_gen",
    ]

    for benchmark_name in BENCHMARKS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"Running: {benchmark_name}")
        print("=" * 60)

        workspace = test_base / benchmark_name
        result = run_benchmark_simple(
            workspace_path=str(workspace),
            model_path_in_docker=model_path_in_docker,
            benchmark_name=benchmark_name,
            gpu_count=GPU_COUNT,
            limit=LIMIT,
            max_error_samples=5,
        )

        error_samples = result.get("error_samples", [])
        benchmark_results = result.get("benchmark_results", [])

        print(f"  Results: {benchmark_results}")
        print(f"  Error samples: {len(error_samples)}")
        if error_samples:
            print(f"  Sample: {error_samples[0]}")

        results_summary[benchmark_name] = {
            "error_count": len(error_samples),
            "benchmark_results": benchmark_results,
        }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, info in results_summary.items():
        print(f"  {name}: errors={info['error_count']}")
