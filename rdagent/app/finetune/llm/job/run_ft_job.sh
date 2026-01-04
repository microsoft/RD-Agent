#!/bin/bash
# Run multiple FT tasks in parallel under a single job directory
#
# Usage: ./run_ft_job.sh [tasks.json]
#
# Config format (tasks.json):
# {
#   "tasks": [
#     {"model": "Qwen/Qwen3-8B", "benchmark": "aime25", "gpus": "0,1"},
#     {"model": "Qwen/Qwen3-8B", "benchmark": "gsm8k", "gpus": "2,3"}
#   ]
# }

set -e

# ========== CONFIG ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDAGENT_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
STAGGER_DELAY=60

usage() {
    echo "Usage: $0 [tasks.json]"
    echo "Run multiple FT tasks under a single job directory."
    echo "UI: streamlit run rdagent/app/finetune/llm/ui/app.py"
    exit 0
}

# ========== PARSE ARGS ==========
CONFIG_FILE=""

for arg in "$@"; do
    case $arg in
        -h|--help) usage ;;
        *) [[ -z "$CONFIG_FILE" ]] && CONFIG_FILE="$arg" ;;
    esac
done

[[ -z "$CONFIG_FILE" ]] && CONFIG_FILE="$SCRIPT_DIR/tasks.json"
[[ ! -f "$CONFIG_FILE" ]] && echo "Error: Config not found: $CONFIG_FILE" && exit 1

# Check .env file
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: .env not found at $ENV_FILE"
    echo "Please create it from template: cp $SCRIPT_DIR/.env.template $ENV_FILE"
    exit 1
fi

# Check jq
command -v jq &>/dev/null || { echo "Error: jq required"; exit 1; }

# ========== SETUP ==========
# Get log and workspace base paths from environment or use defaults
# Default to project-relative paths; can be overridden by environment variables
FT_LOG_BASE="${FT_LOG_BASE:-$RDAGENT_DIR/log}"
FT_WORKSPACE_BASE="${FT_WORKSPACE_BASE:-$RDAGENT_DIR/git_ignore_folder/RD-Agent_workspace}"

JOB_ID=$(date +%Y-%m-%d_%H-%M)
JOB_DIR="$FT_LOG_BASE/$JOB_ID"
if [[ -d "$JOB_DIR" ]]; then
    i=1; while [[ -d "${JOB_DIR}_$i" ]]; do ((i++)); done
    JOB_ID="${JOB_ID}_$i"; JOB_DIR="${JOB_DIR}_$i"
fi
mkdir -p "$JOB_DIR"

cd "$RDAGENT_DIR"

NUM_TASKS=$(jq '.tasks | length' "$CONFIG_FILE")

echo "=============================================="
echo "FT Job: $JOB_ID"
echo "=============================================="
echo "Config:    $CONFIG_FILE"
echo "Tasks:     $NUM_TASKS"
echo "Log:       $JOB_DIR"
echo "Workspace: $FT_WORKSPACE_BASE/$JOB_ID"
echo ""

declare -a PIDS

for ((i=0; i<NUM_TASKS; i++)); do
    model=$(jq -r ".tasks[$i].model" "$CONFIG_FILE")
    benchmark=$(jq -r ".tasks[$i].benchmark" "$CONFIG_FILE")
    gpus=$(jq -r ".tasks[$i].gpus // \"0\"" "$CONFIG_FILE")
    scenario=$(jq -r ".tasks[$i].scenario // empty" "$CONFIG_FILE")
    [[ -z "$scenario" ]] && scenario="Improve model performance on $benchmark"
    model_name=$(basename "$model")
    task_name="${benchmark}_${model_name}"
    trace_path="$JOB_DIR/$task_name"

    echo "Task $i: $task_name (model=$model, benchmark=$benchmark, gpus=$gpus)"

    # Run task
    # Source .env from job directory, then override with task-specific values
    task_workspace="$FT_WORKSPACE_BASE/$JOB_ID/$task_name"
    mkdir -p "$task_workspace"
    (
        set -a && source "$ENV_FILE" && set +a
        export CUDA_VISIBLE_DEVICES="$gpus"
        export LOG_TRACE_PATH="$trace_path"
        export WORKSPACE_PATH="$task_workspace"
        export FT_TARGET_BENCHMARK="$benchmark"
        export FT_USER_TARGET_SCENARIO="$scenario"
        cd "$RDAGENT_DIR"
        python rdagent/app/finetune/llm/loop.py --base-model "$model"
    ) > "$JOB_DIR/${task_name}.log" 2>&1 &

    PIDS+=($!)
    echo "  PID:       ${PIDS[-1]}"
    echo ""

    # Stagger starts
    if [[ $i -eq 0 ]]; then
        # First task: wait for initialization
        # Get FT_FILE_PATH from .env or use default
        FT_FILE_PATH=$(grep -E "^FT_FILE_PATH=" "$ENV_FILE" | cut -d= -f2 | tr -d '"' || echo "")
        [[ -z "$FT_FILE_PATH" ]] && FT_FILE_PATH="$RDAGENT_DIR/git_ignore_folder/finetune"
        DATASET_INFO="$FT_FILE_PATH/datasets/dataset_info.json"

        echo "  Waiting for scenario initialization (dataset_info.json)..."
        while [[ ! -f "$DATASET_INFO" ]]; do
            sleep 5
        done
        echo "  Scenario initialized!"

        echo "  Waiting for llm_finetune conda env..."
        while ! conda run -n llm_finetune python -c "import requests" 2>/dev/null; do
            sleep 10
        done
        echo "  Environment ready!"
    elif [[ $i -lt $((NUM_TASKS - 1)) ]]; then
        sleep $STAGGER_DELAY
    fi
done

echo "=============================================="
echo "All tasks started. PIDs: ${PIDS[*]}"
echo "Monitor: tail -f $JOB_DIR/*.log"
echo "UI: streamlit run rdagent/app/finetune/llm/ui/app.py (Job Folder: $JOB_DIR)"
