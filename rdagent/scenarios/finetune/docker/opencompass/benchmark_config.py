"""
OpenCompass Benchmark Configuration
Dynamic configuration for evaluating fine-tuned models with LoRA adapters.
"""

import os
import sys

from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate

# ============================================================================
# Configuration from Environment Variables
# ============================================================================

BASE_MODEL = os.environ.get("BASE_MODEL")
if not BASE_MODEL:
    raise ValueError("BASE_MODEL environment variable is required")

ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/workspace/output")
NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))
LIMIT = os.environ.get("LIMIT", "")

# ============================================================================
# Task/Dataset Mapping
# ============================================================================

# Map task names to OpenCompass dataset config paths
# Format: 'task_name': ('module.path', 'variable_name')
TASK_DATASET_MAPPING = {
    "aime25": ("aime2025.aime2025_llmjudge_gen_5e9f4f", "aime2025_datasets"),
    "aime2025": ("aime2025.aime2025_llmjudge_gen_5e9f4f", "aime2025_datasets"),
    "aime24": ("aime2024.aime2024_llmjudge_gen_5e9f4f", "aime2024_datasets"),
    "aime2024": ("aime2024.aime2024_llmjudge_gen_5e9f4f", "aime2024_datasets"),
    "gsm8k": ("gsm8k.gsm8k_gen_1d7fe4", "gsm8k_datasets"),
    "math": ("math.math_0shot_gen_393424", "math_datasets"),
    "mmlu": ("mmlu.mmlu_pro_0shot_cot_gen_08c1de", "mmlu_datasets"),
}

# ============================================================================
# Model Configuration
# ============================================================================

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="finetuned_model",
        path=BASE_MODEL,
        peft_path=ADAPTER_PATH,  # Load LoRA adapter using PEFT
        tokenizer_path=BASE_MODEL,
        max_out_len=4096,
        max_seq_len=4096,
        batch_size=8,
        run_cfg=dict(
            num_gpus=NUM_GPUS,
        ),
        generation_kwargs=dict(
            temperature=0.6,
            top_p=0.95,
        ),
    )
]

# ============================================================================
# Dataset Configuration (Dynamic Import)
# ============================================================================

datasets = []
tasks_str = os.environ.get("BENCHMARK_TASKS", "aime25")
tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]

print(f"Configuring datasets for tasks: {tasks}", file=sys.stderr)

# Import datasets dynamically using read_base()
with read_base():
    for task in tasks:
        if task not in TASK_DATASET_MAPPING:
            print(f"Warning: Unknown task '{task}', skipping...", file=sys.stderr)
            continue

        module_path, var_name = TASK_DATASET_MAPPING[task]

        try:
            # Import the dataset configuration
            exec(f"from opencompass.configs.datasets.{module_path} import {var_name}")

            # Get the imported dataset list
            task_datasets = eval(var_name)
            datasets.extend(task_datasets)

            print(f"  ✓ Loaded {task}: {len(task_datasets)} dataset(s)", file=sys.stderr)
        except ImportError as e:
            print(f"  ✗ Failed to import {task} from {module_path}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ Error loading {task}: {e}", file=sys.stderr)

if not datasets:
    raise ValueError(f"No valid datasets loaded from tasks: {tasks}")

print(f"\nTotal datasets configured: {len(datasets)}", file=sys.stderr)

# ============================================================================
# Evaluation Configuration
# ============================================================================

# Apply sample limit if specified (for debugging/testing)
if LIMIT and LIMIT.isdigit():
    limit_value = int(LIMIT)
    print(f"\nApplying sample limit: {limit_value}", file=sys.stderr)

    for dataset in datasets:
        if "reader_cfg" not in dataset:
            dataset["reader_cfg"] = {}
        dataset["reader_cfg"]["test_range"] = f"[:{limit_value}]"
