#!/bin/bash
# LM Evaluation Harness Entrypoint Script
# This script runs inside the Docker container to evaluate fine-tuned models

set -e  # Exit on error

echo "========================================"
echo "LM Evaluation Harness - Benchmark Evaluation"
echo "========================================"

# Read configurations from environment variables
TASKS="${BENCHMARK_TASKS:-gsm8k}"
BASE_MODEL="${BASE_MODEL:?BASE_MODEL environment variable is required}"
ADAPTER_PATH="${ADAPTER_PATH:-/workspace/output}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/benchmark_results}"
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "Configuration:"
echo "  Tasks: ${TASKS}"
echo "  Base Model: ${BASE_MODEL}"
echo "  Adapter Path: ${ADAPTER_PATH}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Limit: ${LIMIT:-unlimited}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Num GPUs: ${NUM_GPUS}"
echo "========================================"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build lm_eval command
CMD="lm_eval --model vllm"
CMD="${CMD} --model_args pretrained=${BASE_MODEL},lora_local_path=${ADAPTER_PATH}"
CMD="${CMD},tensor_parallel_size=${NUM_GPUS},gpu_memory_utilization=0.8"
CMD="${CMD},max_model_len=4096,max_lora_rank=64,dtype=auto,trust_remote_code=True"
CMD="${CMD} --tasks ${TASKS}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --apply_chat_template"
CMD="${CMD} --output_path ${OUTPUT_DIR}/results.json"
CMD="${CMD} --log_samples"

# Add limit if specified
if [ -n "${LIMIT}" ]; then
    CMD="${CMD} --limit ${LIMIT}"
fi

echo ""
echo "Running command:"
echo "${CMD}"
echo ""
echo "========================================"

# Run evaluation
eval ${CMD}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
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

