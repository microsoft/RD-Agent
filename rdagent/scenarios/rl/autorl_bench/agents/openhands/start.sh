#!/bin/bash
# OpenHands Agent wrapper for AutoRL-Bench

echo "=== OpenHands Agent ==="
echo "Task: $TASK"
echo "Model: $BASE_MODEL"
echo "Workspace: $WORKSPACE"
echo "Grading Server: $GRADING_SERVER_URL"
echo "Output Dir: $OUTPUT_DIR"

# Load .env configuration (already in the RD-Agent directory at startup)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env"
fi

# Mapping environment variables (rdagent uses OPENAI_API_KEY, openhands uses LLM_API_KEY)
if [ -z "$OPENAI_API_KEY" ] && [ -n "$LLM_API_KEY" ]; then
    export OPENAI_API_KEY="$LLM_API_KEY"
fi
if [ -z "$LLM_API_KEY" ] && [ -n "$OPENAI_API_KEY" ]; then
    export LLM_API_KEY="$OPENAI_API_KEY"
fi
if [ -z "$LLM_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: LLM_API_KEY or OPENAI_API_KEY required"
    exit 2
fi
# LLM_MODEL is passed in from config.yaml first, otherwise CHAT_MODEL is used, the default is gpt-5
export LLM_MODEL="${LLM_MODEL:-${CHAT_MODEL:-gpt-5}}"
export LLM_BASE_URL="${OPENAI_API_BASE}"
echo "LLM API key length: ${#LLM_API_KEY}"
echo "LLM Model: $LLM_MODEL"

#Training environment Python path (just set TRAINING_PYTHON in .env, no need for conda)
if [ -z "$TRAINING_PYTHON" ]; then
    echo "WARNING: TRAINING_PYTHON not set in .env, trying conda fallback..."
    source "$(conda info --base 2>/dev/null || echo /root/miniconda3)/etc/profile.d/conda.sh" 2>/dev/null
    conda activate "${CONDA_ENV_TRAINING:-autorl}" 2>/dev/null
    export TRAINING_PYTHON="$(which python)"
    conda activate "${CONDA_ENV_OPENHANDS:-openhands}" 2>/dev/null
fi
echo "Training Python: $TRAINING_PYTHON"

# Run openhands-rl pipeline
cd "${OPENHANDS_RL_ROOT:-$HOME/openhands-rl}"
OPENHANDS_PYTHON="${OPENHANDS_PYTHON:-python}"

"$OPENHANDS_PYTHON" main.py \
    --benchmark "$TASK" \
    --base-model "$BASE_MODEL" \
    --workspace "$WORKSPACE" \
    --max-iterations ${MAX_ITERATIONS:-10} \
    --training-timeout ${TRAINING_TIMEOUT:-7200} \
    --max-agent-steps ${MAX_AGENT_STEPS:-50}
