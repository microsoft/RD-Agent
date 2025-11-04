#!/bin/bash
# OpenCompass Benchmark Evaluation Entrypoint Script
# This script runs inside the Docker container to evaluate fine-tuned models using OpenCompass

set -e  # Exit on error

echo "========================================"
echo "OpenCompass - Benchmark Evaluation"
echo "========================================"

# Read configurations from environment variables
TASKS="${BENCHMARK_TASKS:-aime25}"
BASE_MODEL="${BASE_MODEL:?BASE_MODEL environment variable is required}"
ADAPTER_PATH="${ADAPTER_PATH:-/workspace/output}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/benchmark_results}"
LIMIT="${LIMIT:-}"
NUM_GPUS="${NUM_GPUS:-1}"

# Judge API configuration (for llmjudge tasks)
OC_JUDGE_MODEL="${OC_JUDGE_MODEL:-gpt-4}"
OC_JUDGE_API_KEY="${OC_JUDGE_API_KEY:-}"
OC_JUDGE_API_BASE="${OC_JUDGE_API_BASE:-}"

echo "Configuration:"
echo "  Tasks: ${TASKS}"
echo "  Base Model: ${BASE_MODEL}"
echo "  Adapter Path: ${ADAPTER_PATH}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Limit: ${LIMIT:-unlimited}"
echo "  Num GPUs: ${NUM_GPUS}"
echo "  Judge Model: ${OC_JUDGE_MODEL}"
echo "========================================"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p /workspace/configs

# Map task names from lm_eval style to opencompass style
map_task_name() {
    local task=$1
    case "$task" in
        aime25|aime2025)
            echo "aime2025_llmjudge_gen_5e9f4f"
            ;;
        aime24|aime2024)
            echo "aime2024_llmjudge_gen_5e9f4f"
            ;;
        gsm8k)
            echo "gsm8k_gen_1d7fe4"
            ;;
        math)
            echo "math_0shot_gen_393424"
            ;;
        mmlu)
            echo "mmlu_pro_0shot_cot_gen_08c1de"
            ;;
        *)
            # Return original task name if no mapping found
            echo "$task"
            ;;
    esac
}

# Convert comma-separated tasks to opencompass dataset names
IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
OC_DATASETS=()
for task in "${TASK_ARRAY[@]}"; do
    mapped_task=$(map_task_name "$task")
    OC_DATASETS+=("$mapped_task")
done

echo "Mapped datasets: ${OC_DATASETS[*]}"

# Generate dynamic model configuration with LoRA support
cat > /workspace/configs/lora_model.py << EOF
from opencompass.models import VLLMwithChatTemplate

# Dynamic model configuration for fine-tuned model with LoRA adapter
models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='finetuned_model',
        path='${BASE_MODEL}',
        model_kwargs=dict(
            tensor_parallel_size=${NUM_GPUS},
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            dtype='auto',
            enable_lora=True,
            max_lora_rank=64,
        ),
        generation_kwargs=dict(
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
        ),
        max_seq_len=4096,
    )
]
EOF

echo ""
echo "Generated model config at /workspace/configs/lora_model.py"

# Build opencompass command
CMD="opencompass"

# Add datasets
for dataset in "${OC_DATASETS[@]}"; do
    CMD="${CMD} --datasets ${dataset}"
done

# Add model config
CMD="${CMD} --models /workspace/configs/lora_model.py"

# Use vLLM accelerator
CMD="${CMD} -a vllm"

# Add parallelism settings
CMD="${CMD} --max-num-workers ${NUM_GPUS}"
CMD="${CMD} --max-workers-per-gpu 1"

# Add output directory
CMD="${CMD} --work-dir ${OUTPUT_DIR}"

# Add limit if specified
if [ -n "${LIMIT}" ]; then
    CMD="${CMD} --debug --max-num-samples ${LIMIT}"
fi

# Export judge API settings
if [ -n "${OC_JUDGE_API_KEY}" ]; then
    export OC_JUDGE_MODEL="${OC_JUDGE_MODEL}"
    export OC_JUDGE_API_KEY="${OC_JUDGE_API_KEY}"
    export OC_JUDGE_API_BASE="${OC_JUDGE_API_BASE}"
fi

# Set vLLM to use V1 engine
export VLLM_USE_V1=1

echo ""
echo "Running command:"
echo "${CMD}"
echo ""
echo "========================================"

# Run evaluation
eval ${CMD}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    # Convert OpenCompass results to lm_eval compatible format
    python3 << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path
from glob import glob

output_dir = os.environ.get('OUTPUT_DIR', '/workspace/benchmark_results')

# Find OpenCompass summary file
summary_files = glob(f"{output_dir}/**/summary.json", recursive=True)
if not summary_files:
    print("Warning: No summary.json found in output directory")
    # Create empty results
    results = {"results": {}}
else:
    summary_file = summary_files[0]
    print(f"Found summary file: {summary_file}")

    with open(summary_file) as f:
        oc_data = json.load(f)

    # Convert to lm_eval compatible format
    results = {"results": {}}

    # OpenCompass summary format: {dataset_name: {metric: score}}
    for dataset, metrics in oc_data.items():
        if isinstance(metrics, dict):
            # Extract main metric (usually accuracy or score)
            main_score = None
            for metric_name, score in metrics.items():
                if isinstance(score, (int, float)):
                    main_score = score
                    break

            if main_score is not None:
                results["results"][dataset] = {
                    "accuracy": main_score,
                    "accuracy_stderr": 0.0
                }

# Write results to expected location
results_path = f"{output_dir}/results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Converted results written to: {results_path}")
PYTHON_SCRIPT

    echo ""
    echo "========================================"
    echo "Evaluation completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}/results.json"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "ERROR: Evaluation failed with exit code ${EXIT_CODE}"
    echo "========================================"
    exit ${EXIT_CODE}
fi
