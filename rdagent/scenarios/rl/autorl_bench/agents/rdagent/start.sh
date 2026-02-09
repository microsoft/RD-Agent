#!/bin/bash
# RD-Agent wrapper for AutoRL-Bench

echo "=== RD-Agent ==="
echo "Task: $TASK"
echo "Model: $BASE_MODEL"
echo "Workspace: $WORKSPACE"

# 加载 .env 配置（启动时已在 RD-Agent 目录）
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env"
fi

# 设置 rdagent 数据目录（命令行会传 base_model 和 benchmark）
export RL_FILE_PATH=$(dirname $(dirname $MODEL_PATH))
echo "RL_FILE_PATH: $RL_FILE_PATH"

# 运行 rdagent（内部每次迭代会自动调用 grading server 评测）
python -m rdagent.app.rl.loop \
    --base-model "$BASE_MODEL" \
    --benchmark "$TASK" \
    --step-n $STEP_N \
    --loop-n $LOOP_N
