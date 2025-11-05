#!/bin/bash
# OpenCompass Benchmark Evaluation Entrypoint

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

# Judge API configuration (for llmjudge tasks)
OC_JUDGE_MODEL="${OC_JUDGE_MODEL:-gpt-4}"
OC_JUDGE_API_KEY="${OC_JUDGE_API_KEY:-}"
OC_JUDGE_API_BASE="${OC_JUDGE_API_BASE:-}"

mkdir -p "${OUTPUT_DIR}"

# Export judge API settings
if [ -n "${OC_JUDGE_API_KEY}" ]; then
    export OC_JUDGE_MODEL OC_JUDGE_API_KEY OC_JUDGE_API_BASE
fi

export VLLM_USE_V1=1

# ============================================================================
# Run OpenCompass
# ============================================================================

opencompass /app/benchmark_config.py \
    -a vllm \
    --work-dir "${OUTPUT_DIR}" \
    --max-num-workers "${NUM_GPUS}" \
    --max-workers-per-gpu 1

EXIT_CODE=$?

# ============================================================================
# Convert Results to RD-Agent Format
# ============================================================================

if [ ${EXIT_CODE} -eq 0 ]; then
    python3 << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path
from glob import glob

output_dir = os.environ.get('OUTPUT_DIR', '/workspace/benchmark_results')
results = {"results": {}}

# OpenCompass outputs results to: {work_dir}/{timestamp}/results/{model}/{dataset}.json
# Find the latest timestamp directory
result_dirs = sorted(glob(f"{output_dir}/*/results"), reverse=True)

if result_dirs:
    # Get all dataset result files from the latest run
    result_files = glob(f"{result_dirs[0]}/*/*.json")

    for result_file in result_files:
        dataset_name = Path(result_file).stem

        with open(result_file) as f:
            data = json.load(f)

        # Extract accuracy metric
        if 'accuracy' in data:
            accuracy = data['accuracy']
            # Normalize to [0, 1] range
            if accuracy > 1:
                accuracy = accuracy / 100.0

            results["results"][dataset_name] = {
                "accuracy": accuracy,
                "accuracy_stderr": 0.0
            }

# Write results
results_path = f"{output_dir}/results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Converted {len(results['results'])} dataset results")
PYTHON_SCRIPT

    echo "Evaluation completed: ${OUTPUT_DIR}/results.json"
else
    echo "ERROR: Evaluation failed (exit code: ${EXIT_CODE})"
    exit ${EXIT_CODE}
fi
