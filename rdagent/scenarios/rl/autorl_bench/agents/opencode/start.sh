#!/bin/bash
# OpenCode Agent wrapper for AutoRL-Bench

echo "=== OpenCode Agent ==="
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

# opencode-rl 路径：默认用外部独立目录
OPENCODE_RL_ROOT="${OPENCODE_RL_ROOT:-/data/userdata/v-tiansha/opencode-rl}"

# OPENCODE_MODEL 优先从 config.yaml 传入，否则用 CHAT_MODEL，默认 gpt-5
export OPENCODE_MODEL="${OPENCODE_MODEL:-${CHAT_MODEL:-gpt-5}}"
echo "OpenCode Model: $OPENCODE_MODEL"

export PYTHONUNBUFFERED=1

# opencode CLI 可能装在 ~/.opencode/bin，确保在 PATH 中
export PATH="$HOME/.opencode/bin:$PATH"

# 把训练环境的 bin 目录加到 PATH，这样 LLM agent 的 bash 工具调用
# (python3 -c "from trl import ...") 也能用到正确的训练依赖
if [ -n "$TRAINING_PYTHON" ]; then
    TRAINING_BIN_DIR="$(dirname "$TRAINING_PYTHON")"
    export PATH="$TRAINING_BIN_DIR:$PATH"
    echo "Training env bin: $TRAINING_BIN_DIR (prepended to PATH)"
fi

# Python 解释器：优先用 .env 中的 OPENCODE_PYTHON，否则用 python3
PYTHON="${OPENCODE_PYTHON:-python3}"
echo "Python: $PYTHON"

# 生成 opencode config（用 RD-Agent 根 .env 中的 API 配置）
export XDG_CONFIG_HOME="${OPENCODE_RL_ROOT}/.opencode-config"
mkdir -p "$XDG_CONFIG_HOME/opencode"
cat > "$XDG_CONFIG_HOME/opencode/opencode.json" <<EOCFG
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "openai": {
      "npm": "@ai-sdk/openai",
      "name": "Auto-configured",
      "options": {
        "baseURL": "${OPENAI_API_BASE}",
        "apiKey": "${OPENAI_API_KEY}"
      },
      "models": {
        "${OPENCODE_MODEL}": { "name": "${OPENCODE_MODEL}" }
      }
    }
  }
}
EOCFG

# 运行 opencode-rl pipeline
cd "$OPENCODE_RL_ROOT"

# Use exec to REPLACE bash with python3, so signals go directly to python3
# without an intermediate bash process. This avoids double signal delivery.
exec "$PYTHON" main.py \
    --benchmark "$TASK" \
    --base-model "$BASE_MODEL" \
    --run-dir "$WORKSPACE" \
    --max-iterations ${MAX_ITERATIONS:-5} \
    --training-timeout ${TRAINING_TIMEOUT:-7200} \
    --stale-timeout ${STALE_TIMEOUT:-1800} \
    --http-timeout ${HTTP_TIMEOUT:-600} \
    --max-agent-steps ${MAX_AGENT_STEPS:-25}
