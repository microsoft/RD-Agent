#!/bin/bash
# OpenHands Agent wrapper for AutoRL-Bench

echo "=== OpenHands Agent ==="
echo "Task: $TASK"
echo "Model: $BASE_MODEL"
echo "Workspace: $WORKSPACE"
echo "Grading Server: $GRADING_SERVER_URL"
echo "Output Dir: $OUTPUT_DIR"

# 加载 .env 配置（启动时已在 RD-Agent 目录）
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env"
fi

# 映射环境变量（rdagent 用 OPENAI_API_KEY，openhands 用 LLM_API_KEY）
export LLM_API_KEY="${OPENAI_API_KEY}"
# LLM_MODEL 优先从 config.yaml 传入，否则用 CHAT_MODEL，默认 gpt-5
export LLM_MODEL="${LLM_MODEL:-${CHAT_MODEL:-gpt-5}}"
export LLM_BASE_URL="${OPENAI_API_BASE}"
echo "LLM Model: $LLM_MODEL"

# 训练环境 Python 路径（.env 中设 TRAINING_PYTHON 即可，无需 conda）
if [ -z "$TRAINING_PYTHON" ]; then
    echo "WARNING: TRAINING_PYTHON not set in .env, trying conda fallback..."
    source "$(conda info --base 2>/dev/null || echo /root/miniconda3)/etc/profile.d/conda.sh" 2>/dev/null
    conda activate "${CONDA_ENV_TRAINING:-cwy-rl}" 2>/dev/null
    export TRAINING_PYTHON="$(which python)"
    conda activate "${CONDA_ENV_OPENHANDS:-openhands}" 2>/dev/null
fi
echo "Training Python: $TRAINING_PYTHON"

# 运行 openhands-rl pipeline
cd "${OPENHANDS_RL_ROOT:-$HOME/openhands-rl}"
OPENHANDS_PYTHON="${OPENHANDS_PYTHON:-python}"

"$OPENHANDS_PYTHON" main.py \
    --benchmark "$TASK" \
    --base-model "$BASE_MODEL" \
    --workspace "$WORKSPACE" \
    --max-iterations ${MAX_ITERATIONS:-10} \
    --training-timeout ${TRAINING_TIMEOUT:-7200} \
    --max-agent-steps ${MAX_AGENT_STEPS:-50}
