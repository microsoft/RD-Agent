#!/bin/bash
# RD-Agent wrapper for AutoRL-Bench

echo "=== RD-Agent ==="
echo "Task: $TASK"
echo "Model: $BASE_MODEL"
echo "Workspace: $WORKSPACE"

# Load .env configuration (already in the RD-Agent directory at startup)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env"
fi

#Set rdagent data directory (base_model and benchmark will be passed in the command line)
export RL_FILE_PATH=$(dirname $(dirname $MODEL_PATH))
echo "RL_FILE_PATH: $RL_FILE_PATH"

# Run rdagent (grading server evaluation will be automatically called internally for each iteration)
python -m rdagent.app.rl.loop \
    --base-model "$BASE_MODEL" \
    --benchmark "$TASK" \
    --step-n $STEP_N \
    --loop-n $LOOP_N
