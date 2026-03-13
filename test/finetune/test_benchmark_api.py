"""
Standalone test script for API-based benchmark testing.

Usage:
    python test_benchmark_api.py

Uses OpenAI-compatible API with Docker environment for running opencompass.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

# Set FT_file_path BEFORE importing rdagent modules (so Docker mounts correct path)
_project_root = Path(__file__).resolve().parents[2]
os.environ["FT_file_path"] = str(_project_root / "git_ignore_folder" / "finetune_files")

import pandas as pd

from rdagent.components.coder.finetune.conf import get_benchmark_env
from rdagent.scenarios.finetune.benchmark.benchmark import get_benchmark_ranges
from rdagent.scenarios.finetune.benchmark.data.adaptor import BENCHMARK_CONFIG_DICT
from rdagent.scenarios.finetune.benchmark.data.default import extract_error_samples

# OpenCompass API config template
API_CONFIG_TEMPLATE = """
from mmengine.config import read_base
from opencompass.models import OpenAI

# ==================== Dataset Import ====================
with read_base():
{dataset_imports}

# Aggregate all dataset variables
datasets = sum([v for k, v in locals().items() if (k == 'datasets' or k.endswith('_datasets')) and isinstance(v, list)], [])

# Apply dataset modifications
for ds in datasets:
{limit_config}
    pass

# ==================== API Model Configuration ====================
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

models = [
    dict(
        abbr='{model_abbr}',
        type=OpenAI,
        path='{model_path}',
        key='{api_key}',
        openai_api_base='{api_base}',
        meta_template=api_meta_template,
        query_per_second={query_per_second},
        max_out_len={max_out_len},
        max_seq_len={max_seq_len},
        batch_size={batch_size},
        retry={retry},
    ),
]

# ==================== Inference Configuration ====================
infer = dict(
    partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type='LocalRunner',
        max_num_workers={max_num_workers},
        retry=2,
        task=dict(type='OpenICLInferTask'),
    ),
)

# ==================== Evaluation Configuration ====================
eval = dict(
    partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type='LocalRunner',
        max_num_workers=4,
        retry=2,
        task=dict(type='OpenICLEvalTask', dump_details=True),
    ),
)

# ==================== Work Directory ====================
work_dir = '{work_dir}'
"""


def generate_api_config(
    model_abbr: str,
    model_path: str,
    api_key: str,
    api_base: str,
    dataset_imports: list[str],
    limit: int | None = None,
    offset: int = 0,
    test_range: str | None = None,
    work_dir: str = "/workspace",
    max_out_len: int = 8192,
    max_seq_len: int = 32768,
    batch_size: int = 8,
    query_per_second: int = 1,
    max_num_workers: int = 16,
    retry: int = 5,
) -> str:
    """Generate OpenCompass config for API-based model evaluation.

    Args:
        test_range: Direct test_range expression (e.g., "[:min(100, len(index_list)//2)]").
                    If provided, overrides limit/offset parameters.
    """
    # Format dataset imports
    dataset_import_lines = "\n".join(f"    from {module} import *" for module in dataset_imports)

    # Format limit config - support direct test_range or limit/offset
    if test_range:
        # Use direct test_range expression (supports dynamic expressions like len(index_list))
        limit_config = f"""    # Apply test_range for dataset sampling
    if 'reader_cfg' not in ds:
        ds['reader_cfg'] = {{}}
    ds['reader_cfg']['test_range'] = '{test_range}'

    # Sync to evaluator's dataset_cfg
    if 'eval_cfg' in ds and 'evaluator' in ds['eval_cfg']:
        evaluator = ds['eval_cfg']['evaluator']
        if isinstance(evaluator, dict) and 'dataset_cfg' in evaluator:
            if 'reader_cfg' not in evaluator['dataset_cfg']:
                evaluator['dataset_cfg']['reader_cfg'] = {{}}
            evaluator['dataset_cfg']['reader_cfg']['test_range'] = '{test_range}'"""
    elif limit:
        if offset:
            computed_range = f"[{offset}:{offset + limit}]"
        else:
            computed_range = f"[:{limit}]"
        limit_config = f"""    # Limit dataset size for faster testing
    if 'reader_cfg' not in ds:
        ds['reader_cfg'] = {{}}
    ds['reader_cfg']['test_range'] = '{computed_range}'

    # Limit few-shot examples to avoid index out of range
    # FixKRetriever uses fix_id_list to select examples from train/dev split
    if 'infer_cfg' in ds and 'retriever' in ds['infer_cfg']:
        retriever = ds['infer_cfg']['retriever']
        if isinstance(retriever, dict) and 'fix_id_list' in retriever:
            # Limit fix_id_list to valid range (0 to limit-1)
            retriever['fix_id_list'] = [i for i in retriever['fix_id_list'] if i < {limit}]

    # Sync to evaluator's dataset_cfg
    if 'eval_cfg' in ds and 'evaluator' in ds['eval_cfg']:
        evaluator = ds['eval_cfg']['evaluator']
        if isinstance(evaluator, dict) and 'dataset_cfg' in evaluator:
            if 'reader_cfg' not in evaluator['dataset_cfg']:
                evaluator['dataset_cfg']['reader_cfg'] = {{}}
            evaluator['dataset_cfg']['reader_cfg']['test_range'] = '{computed_range}'"""
    else:
        limit_config = ""

    return API_CONFIG_TEMPLATE.format(
        dataset_imports=dataset_import_lines,
        limit_config=limit_config,
        model_abbr=model_abbr,
        model_path=model_path,
        api_key=api_key,
        api_base=api_base,
        work_dir=work_dir,
        max_out_len=max_out_len,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        query_per_second=query_per_second,
        max_num_workers=max_num_workers,
        retry=retry,
    )


def run_benchmark_api(
    workspace_path: str,
    model_name: str,
    api_key: str,
    api_base: str,
    benchmark_name: str,
    limit: int | None = 3,
    offset: int = 0,
    test_range: str | None = None,
    max_error_samples: int = 5,
    max_out_len: int = 8192,
    max_seq_len: int = 32768,
    batch_size: int = 8,
    query_per_second: int = 1,
    max_num_workers: int = 16,
    retry: int = 5,
    hf_token: str | None = None,
    result_subdir: str = "",
):
    """
    API-based benchmark runner using rdagent Docker env.

    Args:
        workspace_path: Local workspace path
        model_name: API model name (e.g., gpt-4o-mini)
        api_key: OpenAI API key
        api_base: OpenAI API base URL (will be converted to Docker-accessible URL)
        benchmark_name: Benchmark name
        limit: Dataset limit (ignored if test_range is provided)
        offset: Starting offset for dataset sampling (ignored if test_range is provided)
        test_range: Direct test_range expression (e.g., "[:min(100, len(index_list)//2)]").
                    If provided, overrides limit/offset parameters.
        max_error_samples: Max error samples to extract
        max_out_len: Maximum output length
        max_seq_len: Maximum sequence length
        batch_size: Batch size for API calls
        query_per_second: Rate limit for API calls
        max_num_workers: Max number of workers for inference
        hf_token: Hugging Face token for gated datasets
        result_subdir: Subdirectory for results (e.g., "validation", "test")
    """
    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)

    cfg = BENCHMARK_CONFIG_DICT[benchmark_name]

    # Auto download dependent data if configured
    if cfg.download is not None:
        cfg.download()

    # Docker uses host network, so localhost works directly
    # OpenAI class (inference) expects full URL with /chat/completions
    docker_api_base = "http://localhost:3000/v1/chat/completions"
    # OpenAISDK class (LLM judge) auto-appends /chat/completions, so use base only
    docker_api_base_sdk = "http://localhost:3000/v1"

    # Generate config.py
    config_content = generate_api_config(
        model_abbr=f"api-{benchmark_name}",
        model_path=model_name,
        api_key=api_key,
        api_base=docker_api_base,
        dataset_imports=[cfg.dataset],
        limit=limit,
        offset=offset,
        test_range=test_range,
        work_dir="/workspace",
        max_out_len=max_out_len,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        query_per_second=query_per_second,
        max_num_workers=max_num_workers,
        retry=retry,
    )

    config_file = workspace / "config.py"
    config_file.write_text(config_content)

    # Get Docker env with cache enabled
    env = get_benchmark_env()
    env.conf.enable_cache = True

    # Environment variables for LLM judge (required for cascade eval benchmarks like AIME25)
    # Note: LLM judge uses OpenAISDK which auto-appends /chat/completions
    env_vars = {
        "OC_JUDGE_MODEL": model_name,
        "OC_JUDGE_API_KEY": api_key,
        "OC_JUDGE_API_BASE": docker_api_base_sdk,  # SDK auto-appends /chat/completions
        "OC_JUDGE_RETRY": "3",
        # Pass API credentials for use inside Docker
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": docker_api_base_sdk,  # SDK auto-appends /chat/completions
    }
    # Add HF token for gated datasets (e.g., ChemCoTBench)
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    # Run opencompass in Docker with --debug to avoid subprocess segfault
    if result_subdir:
        benchmark_work_dir = f"/workspace/benchmark_results/{result_subdir}"
    else:
        benchmark_work_dir = "/workspace/benchmark_results"
    cmd = f"opencompass /workspace/config.py --work-dir {benchmark_work_dir} --debug"
    print(f"Running in Docker: {cmd}")
    print(f"API Base (Docker): {docker_api_base}")
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
    # Get score column (the model name column, e.g., 'api-chemcotbench')
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

    # ==================== API Configuration ====================
    API_KEY = "sk-1234"
    API_BASE = "http://localhost:3000"
    MODEL = "gpt-4o-mini"
    HF_TOKEN = "hf_xxxx"  # For gated datasets

    # ==================== Test Configuration ====================
    MAX_OUT_LEN = 8192
    MAX_SEQ_LEN = 32768
    BATCH_SIZE = 8
    QUERY_PER_SECOND = 1
    MAX_NUM_WORKERS = 16

    # Create test directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_base = _project_root / "git_ignore_folder" / "test_api" / timestamp

    # ==================== Test Mode Selection ====================
    # Set to True to test get_benchmark_ranges() with validation/test splits
    TEST_BENCHMARK_RANGES = True

    if TEST_BENCHMARK_RANGES:
        # Test get_benchmark_ranges() with AIME25 (small dataset, 15 samples per subset)
        val_range, test_range = get_benchmark_ranges()
        print("=" * 60)
        print("TESTING get_benchmark_ranges() NON-OVERLAPPING SPLITS")
        print("=" * 60)
        print(f"Validation range: {val_range}")
        print(f"Test range: {test_range}")
        print(f"API Base: {API_BASE}")
        print(f"Output: {test_base}")
        print("=" * 60)

        # Test with AIME25 - a small dataset (15 samples per subset)
        BENCHMARK = "aime25"
        results_summary = {}

        for split_name, split_range in [("validation", val_range), ("test", test_range)]:
            print(f"\n{'='*60}")
            print(f"Running: {BENCHMARK} - {split_name} split")
            print(f"test_range: {split_range}")
            print("=" * 60)

            workspace = test_base / BENCHMARK / split_name
            result = run_benchmark_api(
                workspace_path=str(workspace),
                model_name=MODEL,
                api_key=API_KEY,
                api_base=API_BASE,
                benchmark_name=BENCHMARK,
                limit=None,  # Disabled, use test_range instead
                test_range=split_range,
                max_error_samples=5,
                max_out_len=MAX_OUT_LEN,
                max_seq_len=MAX_SEQ_LEN,
                batch_size=BATCH_SIZE,
                query_per_second=QUERY_PER_SECOND,
                max_num_workers=MAX_NUM_WORKERS,
                hf_token=HF_TOKEN,
                result_subdir=split_name,
            )

            error_samples = result.get("error_samples", [])
            benchmark_results = result.get("benchmark_results", {})

            # Save result to workspace
            result_file = workspace / "result.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Result saved to: {result_file}")

            print(f"  Results: {benchmark_results}")
            print(f"  Error samples: {len(error_samples)}")

            results_summary[f"{BENCHMARK}_{split_name}"] = {
                "error_count": len(error_samples),
                "benchmark_results": benchmark_results,
            }

        print("\n" + "=" * 60)
        print("SUMMARY - get_benchmark_ranges() TEST")
        print("=" * 60)
        for name, info in results_summary.items():
            print(f"  {name}: {info['benchmark_results']}")

    else:
        # Original test mode with fixed limit/offset
        LIMIT = 3
        print("=" * 60)
        print(f"API BENCHMARK TEST: {MODEL} (limit={LIMIT})")
        print(f"API Base: {API_BASE}")
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
            "panorama",
            "panorama_par4pc",
            "panorama_pi4pc",
            "panorama_noc4pc",
            # PANORAMA - Patent Analysis (CoT)
            "panorama_par4pc_cot",
            "panorama_pi4pc_cot",
            "panorama_noc4pc_cot",
            # ChemCoTBench - Chemistry Reasoning
            "chemcotbench",
            "chemcotbench_mol_und",
            "chemcotbench_mol_edit",
            "chemcotbench_mol_opt",
            "chemcotbench_reaction",
            # TableBench - Table QA
            "tablebench_data_analysis",
            "tablebench_fact_checking",
            "tablebench_numerical_reasoning",
            "tablebench_visualization",
            "tablebench_gen",
            # Finance
            "FinanceIQ_gen",
        ]

        for benchmark_name in BENCHMARKS_TO_TEST:
            print(f"\n{'='*60}")
            print(f"Running: {benchmark_name}")
            print("=" * 60)

            workspace = test_base / benchmark_name
            result = run_benchmark_api(
                workspace_path=str(workspace),
                model_name=MODEL,
                api_key=API_KEY,
                api_base=API_BASE,
                benchmark_name=benchmark_name,
                limit=LIMIT,
                max_error_samples=5,
                max_out_len=MAX_OUT_LEN,
                max_seq_len=MAX_SEQ_LEN,
                batch_size=BATCH_SIZE,
                query_per_second=QUERY_PER_SECOND,
                max_num_workers=MAX_NUM_WORKERS,
                hf_token=HF_TOKEN,
                offset=100,
            )

            error_samples = result.get("error_samples", [])
            benchmark_results = result.get("benchmark_results", [])

            # Save result to workspace
            result_file = workspace / "result.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Result saved to: {result_file}")

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
