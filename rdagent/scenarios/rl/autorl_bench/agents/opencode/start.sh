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
exec python3 main.py \
    --benchmark "$TASK" \
    --base-model "$BASE_MODEL" \
    --max-iterations ${MAX_ITERATIONS:-5} \
    --max-retries ${MAX_RETRIES:-20} \
    --training-timeout ${TRAINING_TIMEOUT:-7200} \
    --stale-timeout ${STALE_TIMEOUT:-1800} \
    --http-timeout ${HTTP_TIMEOUT:-600} \
    --eval-timeout ${EVAL_TIMEOUT:-7200} \
    --max-agent-steps ${MAX_AGENT_STEPS:-25}
