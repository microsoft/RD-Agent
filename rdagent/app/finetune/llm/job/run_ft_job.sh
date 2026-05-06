#!/bin/bash
# Run multiple FT tasks in parallel under a single job directory
#
# Usage: ./run_ft_job.sh [tasks.json]
#
# Config format (tasks.json):
# {
#   "tasks": [
#     {"model": "Qwen/Qwen2.5-7B-Instruct", "benchmark": "aime25", "gpus": "0,1"},
#     {"model": "Qwen/Qwen2.5-7B-Instruct", "benchmark": "chemcotbench_mol_edit", "gpus": "2,3"}
#   ]
# }

set -e

# ========== CONFIG ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDAGENT_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
SCENARIOS_FILE="$SCRIPT_DIR/scenarios.json"
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
    echo "Create it from your repository root config, for example:"
    echo "  cp $RDAGENT_DIR/.env $ENV_FILE"
    exit 1
fi

# Check jq
command -v jq &>/dev/null || { echo "Error: jq required"; exit 1; }
command -v tmux &>/dev/null || { echo "Error: tmux required"; exit 1; }

# Read the FT runtime mode from the job env file for readiness checks.
FT_ENV_TYPE=$(grep -E "^FT_Coder_CoSTEER_env_type=" "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d "\"'" || true)
FT_ENV_TYPE="${FT_ENV_TYPE:-docker}"

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
echo "FT env:    $FT_ENV_TYPE"
echo ""

# Setup tmux session
TMUX_SESSION="rdagent"
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" -n "main"
echo "Tmux session created: $TMUX_SESSION"
echo ""

for ((i=0; i<NUM_TASKS; i++)); do
    model=$(jq -r ".tasks[$i].model" "$CONFIG_FILE")
    benchmark=$(jq -r ".tasks[$i].benchmark" "$CONFIG_FILE")
    gpus=$(jq -r ".tasks[$i].gpus // \"0\"" "$CONFIG_FILE")
    port=$(jq -r ".tasks[$i].port // empty" "$CONFIG_FILE")
    task_timeout=$(jq -r ".tasks[$i].timeout // \"12h\"" "$CONFIG_FILE")

    # Load benchmark_description: tasks.json -> scenarios.json
    benchmark_desc=$(jq -r ".tasks[$i].benchmark_description // empty" "$CONFIG_FILE")
    if [[ -z "$benchmark_desc" ]]; then
        benchmark_desc=$(jq -r ".[\"$benchmark\"].benchmark_description // empty" "$SCENARIOS_FILE")
    fi
    if [[ -z "$benchmark_desc" ]]; then
        echo "Error: missing benchmark_description for benchmark '$benchmark'."
        echo "Add it to the task entry in $CONFIG_FILE or register it in $SCENARIOS_FILE."
        exit 1
    fi
    # Note: Special characters in benchmark_desc are handled by writing to env file
    model_name=$(basename "$model")
    task_name="${benchmark}_${model_name}"
    trace_path="$JOB_DIR/$task_name"

    port_info=""
    [[ -n "$port" ]] && port_info=", port=$port"
    echo "Task $i: $task_name (model=$model, benchmark=$benchmark, gpus=$gpus$port_info)"

    # Run task in tmux window with script -c for output capture
    task_workspace="$FT_WORKSPACE_BASE/$JOB_ID/$task_name"
    mkdir -p "$task_workspace"
    LOG_FILE="$JOB_DIR/${task_name}.log"

    # Write task-specific env file (avoids command-line escaping issues with special chars)
    TASK_ENV_FILE="$task_workspace/.task_env"
    cat > "$TASK_ENV_FILE" << EOF
CUDA_VISIBLE_DEVICES='$gpus'
LOG_TRACE_PATH='$trace_path'
WORKSPACE_PATH='$task_workspace'
FT_TARGET_BENCHMARK='$benchmark'
EOF
    if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
        printf "CONDA_ENV_NAME=%q\n" "$CONDA_ENV_NAME" >> "$TASK_ENV_FILE"
    fi
    # Escape shell special characters for double-quoted string: \ " ` $
    if [[ -n "$benchmark_desc" ]]; then
        escaped_desc="$benchmark_desc"
        escaped_desc="${escaped_desc//\\/\\\\}"  # \ -> \\
        escaped_desc="${escaped_desc//\"/\\\"}"  # " -> \"
        escaped_desc="${escaped_desc//\`/\\\`}"  # ` -> \`
        escaped_desc="${escaped_desc//\$/\\\$}"  # $ -> \$
        echo "FT_BENCHMARK_DESCRIPTION=\"$escaped_desc\"" >> "$TASK_ENV_FILE"
    fi
    [[ -n "$port" ]] && echo "OPENAI_API_BASE='http://localhost:$port'" >> "$TASK_ENV_FILE"

    # Create tmux window for this task and get its full target (e.g., rdagent:1.0)
    # Use "session:" format to ensure window is created in the correct session
    WIN_TARGET=$(tmux new-window -t "$TMUX_SESSION:" -n "$benchmark" -P)

    # Build the command with environment setup (env vars loaded from file)
    timeout_arg=""
    [[ -n "$task_timeout" ]] && timeout_arg="--timeout $task_timeout"

    TASK_RUN_SCRIPT="$task_workspace/run_task.sh"
    cat > "$TASK_RUN_SCRIPT" << EOF
#!/bin/bash
set -e
set -a
source "$ENV_FILE"
source "$TASK_ENV_FILE"
set +a

CONDA_ENV_NAME="\${CONDA_ENV_NAME:-rdagent}"
if command -v conda >/dev/null 2>&1; then
    eval "\$(conda shell.bash hook)"
elif [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "\$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: conda not found"
    exit 1
fi

conda activate "\$CONDA_ENV_NAME"
cd "$RDAGENT_DIR"
python rdagent/app/finetune/llm/loop.py --base-model "$model" $timeout_arg
EOF
    chmod +x "$TASK_RUN_SCRIPT"

    # Run with script -c to capture terminal output (using full target for reliability)
    tmux send-keys -t "$WIN_TARGET" "script -q '$LOG_FILE' -c 'bash \"$TASK_RUN_SCRIPT\"'" Enter

    echo "  Window:    $benchmark"
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

        if [[ "$FT_ENV_TYPE" == "conda" ]]; then
            echo "  Waiting for llm_finetune conda env..."
            while ! conda run -n llm_finetune python -c "import requests" 2>/dev/null; do
                sleep 10
            done

            echo "  Waiting for opencompass conda env..."
            while ! conda run -n opencompass python -c "import opencompass" 2>/dev/null; do
                sleep 10
            done
            echo "  Conda environments ready!"
        else
            echo "  Docker mode detected; skipping conda environment readiness checks."
        fi
    elif [[ $i -lt $((NUM_TASKS - 1)) ]]; then
        sleep $STAGGER_DELAY
    fi
done

echo "=============================================="
echo "All tasks started in tmux session: $TMUX_SESSION"
echo "  - Attach:  tmux attach -t $TMUX_SESSION"
echo "  - List:    tmux list-windows -t $TMUX_SESSION"
echo "  - Select:  tmux select-window -t $TMUX_SESSION:{window_name}"
echo "Monitor: tail -f $JOB_DIR/*.log"
echo "UI: streamlit run rdagent/app/finetune/llm/ui/app.py (Job Folder: $JOB_DIR)"
