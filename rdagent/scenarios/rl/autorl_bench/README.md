# AutoRL-Bench

LLM RL Post-Training 评测基准。

## 快速开始

```bash
cd /path/to/RD-Agent
pip install -e .
```

### 运行 Agent

```bash
# OpenHands Agent
python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent openhands \
    --task gsm8k \
    --model Qwen/Qwen2.5-0.5B \
    --timeout 7200

# RD-Agent
python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent rdagent \
    --task gsm8k \
    --model Qwen/Qwen2.5-0.5B \
    --timeout 14400
```

## 目录结构

```
autorl_bench/
├── run.py                    # 入口脚本
├── conf.py                   # 路径配置
│
├── core/                     # 【主干代码】不需要修改
│   ├── __init__.py
│   ├── evaluator.py          # BaseEvaluator 基类
│   ├── opencompass.py        # OpenCompassEvaluator（通用）
│   ├── server.py             # Grading Server
│   ├── utils.py              # 工具函数
│   └── instructions.txt      # Agent 使用说明
│
├── benchmarks/               # 【Benchmark 扩展】添加新任务
│   ├── __init__.py           # 注册表 BENCHMARKS
│   ├── gsm8k/
│   │   └── description.md    # OpenCompass 类只需配置
│   └── alfworld/
│       ├── eval.py           # 自定义评测需要 eval.py
│       └── description.md
│
├── agents/                   # 【Agent 扩展】添加新 Agent
│   ├── __init__.py           # 注册表 AGENTS
│   ├── openhands/
│   │   ├── start.sh
│   │   └── config.yaml
│   └── rdagent/
│       └── ...
│
├── workspace/                # [运行时] 工作目录
├── results/                  # [运行时] 结果
└── log/                      # [运行时] 日志
```

## 核心接口

### BaseEvaluator

所有 benchmark 评测器的基类：

```python
from rdagent.scenarios.rl.autorl_bench.core import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    def run_eval(self, model_path: str, workspace_path: str, **kwargs) -> dict:
        """
        返回: {"score": float, "accuracy_summary": dict, ...}
        """
        pass
```

### Grading Server API

| Endpoint | Method | 说明 |
|----------|--------|------|
| `/submit` | POST | 提交模型评测，返回 score + best |
| `/health` | GET | 健康检查 |
| `/set_baseline` | POST | 设置 baseline 分数 |

### submit 响应格式

```json
{
  "submission_id": 1,
  "score": 65.0,
  "baseline_score": 45.0,
  "improvement": 20.0,
  "best": {"submission_id": 1, "score": 65.0, ...},
  "total_submissions": 5
}
```

## 添加新 Benchmark

### 方式一：使用 OpenCompass（推荐）

只需配置，不需要写代码：

```bash
# 1. 创建目录
mkdir -p benchmarks/new_task/

# 2. 添加描述文件
echo "# New Task\n## 目标\n..." > benchmarks/new_task/description.md

# 3. 注册（benchmarks/__init__.py）
```

```python
BENCHMARKS["new_task"] = BenchmarkConfig(
    id="new_task",
    evaluator_class="rdagent.scenarios.rl.autorl_bench.core.opencompass.OpenCompassEvaluator",
    data_source="huggingface/dataset_name",
    description="New task description",
    eval_config={
        "dataset": "opencompass.configs.datasets.xxx",
    },
)
```

### 方式二：自定义评测

需要实现 eval.py：

```bash
# 1. 创建目录
mkdir -p benchmarks/new_task/
```

```python
# 2. 实现评测器 (benchmarks/new_task/eval.py)
from rdagent.scenarios.rl.autorl_bench.core import BaseEvaluator

class NewTaskEvaluator(BaseEvaluator):
    def __init__(self, config):
        self.config = config
    
    def run_eval(self, model_path: str, workspace_path: str, **kwargs) -> dict:
        # 自定义评测逻辑
        return {"score": 85.0, "accuracy_summary": {...}}
```

```python
# 3. 注册（benchmarks/__init__.py）
BENCHMARKS["new_task"] = BenchmarkConfig(
    id="new_task",
    evaluator_class="rdagent.scenarios.rl.autorl_bench.benchmarks.new_task.eval.NewTaskEvaluator",
    data_source="...",
    ...
)
```

### 添加描述文件

创建 `benchmarks/new_task/description.md`，描述任务目标、数据格式、评测指标。

## 添加新 Agent

### 1. 创建目录

```bash
mkdir -p agents/my_agent/
```

### 2. 创建启动脚本

```bash
# agents/my_agent/start.sh
#!/bin/bash
echo "Task: $TASK"
echo "Model: $BASE_MODEL"
echo "Workspace: $WORKSPACE"

# 你的训练逻辑
python train.py --model $MODEL_PATH --data $DATA_PATH --output $OUTPUT_DIR

# 提交评测
curl -X POST $GRADING_SERVER_URL/submit -H "Content-Type: application/json" \
    -d '{"model_path": "'$OUTPUT_DIR'"}'
```

### 3. 注册 Agent

```python
# agents/__init__.py
AGENTS["my_agent"] = AgentConfig(
    id="my_agent",
    name="My Agent",
    start=get_agents_dir() / "my_agent" / "start.sh",
)
```

## 环境变量

Agent 启动时可用的环境变量：

| 变量 | 说明 |
|------|------|
| `TASK` | 任务名称 (gsm8k, math, ...) |
| `BASE_MODEL` | 基础模型名称 |
| `WORKSPACE` | 工作目录 |
| `MODEL_PATH` | 模型路径 |
| `DATA_PATH` | 数据路径 |
| `OUTPUT_DIR` | 输出目录 |
| `GRADING_SERVER_URL` | 评分服务器 URL |
