#!/bin/bash
# OpenCompass Benchmark Evaluation Entrypoint for Fine-tuned Models

set -e

# ============================================================================
# Configuration
# ============================================================================

: "${BASE_MODEL:?ERROR: BASE_MODEL environment variable is required}"

BENCHMARK_TASKS="${BENCHMARK_TASKS:-aime25}"
ADAPTER_PATH="${ADAPTER_PATH:-/workspace/output}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/benchmark_results}"
NUM_GPUS="${NUM_GPUS:-1}"
LIMIT="${LIMIT:-}"

# Judge API configuration (for cascade evaluation like AIME2025)
OC_JUDGE_MODEL="${OC_JUDGE_MODEL:-gpt-4}"
OC_JUDGE_API_KEY="${OC_JUDGE_API_KEY:-}"
OC_JUDGE_API_BASE="${OC_JUDGE_API_BASE:-}"

mkdir -p "${OUTPUT_DIR}"

# Export judge API settings for config generator
export OC_JUDGE_MODEL OC_JUDGE_API_KEY OC_JUDGE_API_BASE
export BASE_MODEL ADAPTER_PATH NUM_GPUS LIMIT

# Enable vLLM v1 API
export VLLM_USE_V1=1

# ============================================================================
# Run OpenCompass for Each Benchmark
# ============================================================================

echo "=========================================="
echo "OpenCompass Fine-tuned Model Evaluation"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Adapter: ${ADAPTER_PATH}"
echo "Tasks: ${BENCHMARK_TASKS}"
echo "GPUs: ${NUM_GPUS}"
if [ -n "${LIMIT}" ]; then
    echo "Test Limit: ${LIMIT}"
fi
echo "=========================================="
echo ""

# Split tasks by comma
IFS=',' read -ra TASKS <<< "${BENCHMARK_TASKS}"

OVERALL_EXIT_CODE=0
SUCCESSFUL_TASKS=()
FAILED_TASKS=()

for TASK in "${TASKS[@]}"; do
    TASK=$(echo "$TASK" | xargs)  # Trim whitespace

    if [ -z "$TASK" ]; then
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Running benchmark: ${TASK}"
    echo "=========================================="

    # Generate configuration for this task
    export BENCHMARK_TASK="${TASK}"
    TEMP_CONFIG="/tmp/oc_config_${TASK}.py"

    echo "Generating configuration..."
    if ! python3 /app/generate_benchmark_config.py > "${TEMP_CONFIG}"; then
        echo "ERROR: Failed to generate config for ${TASK}"
        FAILED_TASKS+=("${TASK}")
        OVERALL_EXIT_CODE=1
        continue
    fi

    TASK_OUTPUT_DIR="${OUTPUT_DIR}/${TASK}"
    mkdir -p "${TASK_OUTPUT_DIR}"

    # Run OpenCompass evaluation
    echo "Starting OpenCompass evaluation..."
    if opencompass "${TEMP_CONFIG}" \
        -a vllm \
        --work-dir "${TASK_OUTPUT_DIR}" \
        --max-num-workers "${NUM_GPUS}" \
        --max-workers-per-gpu 1; then

        echo "✓ Benchmark ${TASK} completed successfully"
        SUCCESSFUL_TASKS+=("${TASK}")
    else
        echo "✗ Benchmark ${TASK} failed"
        FAILED_TASKS+=("${TASK}")
        OVERALL_EXIT_CODE=1
    fi

    # Clean up temp config
    rm -f "${TEMP_CONFIG}"
done

# ============================================================================
# Aggregate Results
# ============================================================================

echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "Successful: ${#SUCCESSFUL_TASKS[@]}"
for task in "${SUCCESSFUL_TASKS[@]}"; do
    echo "  ✓ ${task}"
done

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo "Failed: ${#FAILED_TASKS[@]}"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  ✗ ${task}"
    done
fi
echo "=========================================="

# ============================================================================
# Convert Results to RD-Agent Format
# ============================================================================

if [ ${#SUCCESSFUL_TASKS[@]} -gt 0 ]; then
    echo ""
    echo "Converting results to RD-Agent format..."

    python3 << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path
from glob import glob

output_dir = os.environ.get('OUTPUT_DIR', '/workspace/benchmark_results')
aggregated_results = {"results": {}}

# Process each task directory
for task_dir in Path(output_dir).iterdir():
    if not task_dir.is_dir():
        continue

    task_name = task_dir.name

    # Find latest result directory: {work_dir}/{timestamp}/results
    result_dirs = sorted(glob(f"{task_dir}/*/results"), reverse=True)

    if not result_dirs:
        print(f"Warning: No results found for {task_name}")
        continue

    # Get all dataset result files from the latest run
    result_files = glob(f"{result_dirs[0]}/*/*.json")

    for result_file in result_files:
        dataset_name = Path(result_file).stem

        try:
            with open(result_file) as f:
                data = json.load(f)

            # Extract accuracy metric
            if 'accuracy' in data:
                accuracy = data['accuracy']
                # Normalize to [0, 1] range
                if accuracy > 1:
                    accuracy = accuracy / 100.0

                aggregated_results["results"][f"{task_name}_{dataset_name}"] = {
                    "accuracy": accuracy,
                    "accuracy_stderr": 0.0
                }
        except Exception as e:
            print(f"Warning: Failed to parse {result_file}: {e}")

# Write aggregated results
results_path = f"{output_dir}/results.json"
with open(results_path, 'w') as f:
    json.dump(aggregated_results, f, indent=2)

print(f"✓ Converted {len(aggregated_results['results'])} dataset results")
print(f"Results saved to: {results_path}")
PYTHON_SCRIPT

    echo "Evaluation completed: ${OUTPUT_DIR}/results.json"
fi

exit ${OVERALL_EXIT_CODE}
