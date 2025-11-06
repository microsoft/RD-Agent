#!/usr/bin/env python3
"""
Generate OpenCompass benchmark configuration for fine-tuned models.
Supports both rule-based and cascade evaluation with LoRA adapters.
"""

import json
import os
import sys

# Benchmark configurations
BENCHMARKS = {
    "aime24": {
        "name": "AIME 2024",
        "import_path": "opencompass.configs.datasets.aime2024.aime2024_gen_17d799",
        "datasets_var": "aime2024_datasets",
        "eval_type": "rule",
    },
    "aime25": {
        "name": "AIME 2025",
        "import_path": "opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f",
        "datasets_var": "aime2025_datasets",
        "eval_type": "cascade",
        "cascade_imports": [
            "aime2025_reader_cfg",
            "aime2025_infer_cfg",
            "GRADER_TEMPLATE",
        ],
    },
    "aime2025": {
        "name": "AIME 2025",
        "import_path": "opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f",
        "datasets_var": "aime2025_datasets",
        "eval_type": "cascade",
        "cascade_imports": [
            "aime2025_reader_cfg",
            "aime2025_infer_cfg",
            "GRADER_TEMPLATE",
        ],
    },
    "gsm8k": {
        "name": "GSM8K",
        "import_path": "opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4",
        "datasets_var": "gsm8k_datasets",
        "eval_type": "rule",
    },
    "math": {
        "name": "MATH",
        "import_path": "opencompass.configs.datasets.math.math_0shot_gen_393424",
        "datasets_var": "math_datasets",
        "eval_type": "rule",
    },
    "mmlu": {
        "name": "MMLU",
        "import_path": "opencompass.configs.datasets.mmlu.mmlu_gen",
        "datasets_var": "mmlu_datasets",
        "eval_type": "rule",
    },
}


def get_model_config(base_model, adapter_path, num_gpus):
    """Generate model configuration with vLLM + LoRA adapter support."""
    return f"""    dict(
        type=VLLM,
        abbr='finetuned-model',
        path='{base_model}',
        lora_path='{adapter_path}',
        model_kwargs=dict(
            tensor_parallel_size={num_gpus},
            trust_remote_code=True,
            dtype='bfloat16',
            gpu_memory_utilization=0.9,
            enable_lora=True,
            max_lora_rank=64,
        ),
        max_seq_len=32768,
        max_out_len=8192,
        batch_size=16,
        generation_kwargs=dict(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        ),
        use_fastchat_template=True,
        run_cfg=dict(num_gpus={num_gpus}, num_procs=1),
    )"""


def generate_rule_based_config(benchmark_cfg, base_model, adapter_path, num_gpus, limit=None):
    """Generate configuration for rule-based evaluation."""
    config = f"""from mmengine.config import read_base
from opencompass.models import VLLM

with read_base():
    from {benchmark_cfg['import_path']} import {benchmark_cfg['datasets_var']}

models = [
{get_model_config(base_model, adapter_path, num_gpus)}
]

datasets = {benchmark_cfg['datasets_var']}
"""

    if limit:
        config += f"""
# Apply test limit
for dataset in datasets:
    if 'reader_cfg' not in dataset:
        dataset['reader_cfg'] = {{}}
    dataset['reader_cfg']['test_range'] = '[:{limit}]'
"""

    return config


def generate_cascade_config(benchmark_cfg, base_model, adapter_path, num_gpus, limit=None):
    """Generate configuration for cascade evaluation (AIME2025)."""
    imports = ", ".join(benchmark_cfg["cascade_imports"])

    # Get judge API configuration from environment
    judge_model = os.environ.get("OC_JUDGE_MODEL", "gpt-5")
    judge_api_key = os.environ.get("OC_JUDGE_API_KEY", "")
    judge_api_base = os.environ.get("OC_JUDGE_API_BASE", "")

    config = f"""from mmengine.config import read_base
from opencompass.models import VLLM
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.datasets import CustomDataset, generic_llmjudge_postprocess
from opencompass.evaluator import CascadeEvaluator, GenericLLMEvaluator, MATHVerifyEvaluator

with read_base():
    from {benchmark_cfg['import_path']} import (
        {imports}
    )

models = [
{get_model_config(base_model, adapter_path, num_gpus)}
]

# Cascade evaluator: rule-based (MATHVerify) + LLM judge fallback
cascade_evaluator = dict(
    type=CascadeEvaluator,
    rule_evaluator=dict(type=MATHVerifyEvaluator),
    llm_evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
                )],
                round=[dict(role='HUMAN', prompt=GRADER_TEMPLATE)],
            ),
        ),
        dataset_cfg=dict(
            type=CustomDataset,
            path='opencompass/aime2025',
            reader_cfg=aime2025_reader_cfg,
        ),
        judge_cfg=dict(
            type='opencompass.models.OpenAISDK',
            path='{judge_model}',
            key='{judge_api_key}',
            openai_api_base=['{judge_api_base}'],
            meta_template=dict(round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ]),
            query_per_second=16,
            batch_size=1024,
            temperature=0.001,
            max_out_len=16384,
            max_seq_len=200000,
        ),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
    parallel=False,
)

{benchmark_cfg['datasets_var']} = [
    dict(
        type=CustomDataset,
        abbr='aime2025',
        path='opencompass/aime2025',
        reader_cfg=aime2025_reader_cfg,
        infer_cfg=aime2025_infer_cfg,
        eval_cfg=dict(evaluator=cascade_evaluator),
    )
]

datasets = {benchmark_cfg['datasets_var']}
"""

    if limit:
        config += f"""
# Apply test limit
datasets[0]['reader_cfg']['test_range'] = '[:{limit}]'
"""

    return config


def main():
    # Get configuration from environment
    base_model = os.environ.get("BASE_MODEL")
    adapter_path = os.environ.get("ADAPTER_PATH", "/workspace/output")
    num_gpus = int(os.environ.get("NUM_GPUS", "1"))
    limit = os.environ.get("LIMIT", "")
    limit_value = int(limit) if limit and limit.isdigit() else None

    if not base_model:
        print("ERROR: BASE_MODEL environment variable is required", file=sys.stderr)
        sys.exit(1)

    # Get benchmark task
    task = os.environ.get("BENCHMARK_TASK", "aime25")

    if task not in BENCHMARKS:
        print(f"ERROR: Unknown benchmark '{task}'", file=sys.stderr)
        print(f"Available: {', '.join(BENCHMARKS.keys())}", file=sys.stderr)
        sys.exit(1)

    benchmark_cfg = BENCHMARKS[task]
    eval_type = benchmark_cfg["eval_type"]

    print(f"# Generated config for {benchmark_cfg['name']}", file=sys.stderr)
    print(f"# Eval type: {eval_type}", file=sys.stderr)
    print(f"# Base model: {base_model}", file=sys.stderr)
    print(f"# Adapter: {adapter_path}", file=sys.stderr)
    print(f"# GPUs: {num_gpus}", file=sys.stderr)
    if limit_value:
        print(f"# Test limit: {limit_value}", file=sys.stderr)
    print("", file=sys.stderr)

    # Generate configuration based on eval type
    if eval_type == "cascade":
        config = generate_cascade_config(benchmark_cfg, base_model, adapter_path, num_gpus, limit_value)
    else:
        config = generate_rule_based_config(benchmark_cfg, base_model, adapter_path, num_gpus, limit_value)

    print(config)


if __name__ == "__main__":
    main()
