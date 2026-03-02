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

# 记录训练环境（cwy-rl）的 Python 路径，供 main.py 执行 train.py 时使用
source "$(conda info --base)/etc/profile.d/conda.sh"
TRAINING_ENV="${CONDA_ENV_TRAINING:-cwy-rl}"
conda activate "$TRAINING_ENV"
export TRAINING_PYTHON="$(which python)"
echo "Training env: $TRAINING_ENV (python=$TRAINING_PYTHON)"

# 切到 openhands 环境运行 agent
conda activate "${CONDA_ENV_OPENHANDS:-openhands}"

# 运行 openhands-rl pipeline（从 .env 读取路径）
cd "${OPENHANDS_RL_ROOT:-$HOME/openhands-rl}"

python main.py \
    --benchmark "$TASK" \
    --base-model "$BASE_MODEL" \
    --workspace "$WORKSPACE" \
    --max-iterations ${MAX_ITERATIONS:-5} \
    --training-timeout ${TRAINING_TIMEOUT:-3600} \
    --max-agent-steps ${MAX_AGENT_STEPS:-25}
