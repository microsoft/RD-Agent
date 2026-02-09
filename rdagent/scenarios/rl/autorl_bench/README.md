# AutoRL-Bench

LLM RL Post-Training 评测基准。

## 快速开始

```bash
cd /Data/home/v-wanyichen/cwy/program/RD-Agent
pip install -e .
```

### Example Agent（GRPO 训练）

```bash
nohup python -m rdagent.scenarios.rl.autorl_bench.run_agent \
    --agent-id example_agent \
    --task gsm8k \
    --base-model Qwen/Qwen2.5-Coder-7B \
    --timeout 7200 \
    > rdagent/scenarios/rl/autorl_bench/log/example_agent.log 2>&1 &

tail -f rdagent/scenarios/rl/autorl_bench/log/example_agent.log
kill $(pgrep -f "run_agent.*example_agent")
```

### RD-Agent（自动假设 + 代码生成）

```bash
nohup python -m rdagent.scenarios.rl.autorl_bench.run_agent \
    --agent-id rdagent \
    --task gsm8k \
    --base-model Qwen/Qwen2.5-Coder-7B \
    --timeout 14400 \
    > rdagent/scenarios/rl/autorl_bench/log/rdagent.log 2>&1 &

tail -f rdagent/scenarios/rl/autorl_bench/log/rdagent.log
kill $(pgrep -f "run_agent.*rdagent")
```

## 目录结构

```
autorl_bench/
├── run_agent.py              # 入口
├── conf.py                   # 配置
├── agents/                   # Agent 实现
│   ├── registry.py
│   ├── example_agent/        # 示例（GRPO）
│   └── rdagent/              # RD-Agent
├── benchmark/                # 评测逻辑（OpenCompass）
│   └── benchmark.py
├── environment/              # Grading Server
│   └── grading_server.py
├── tasks/                    # 任务注册（gsm8k, math...）
│   └── __init__.py
├── utils/
│   ├── download.py           # 下载（模型/数据）
│   └── grading.py            # Grading 客户端
├── workspace/                # [运行时] 见下方
├── results/                  # [运行时] 结果
└── log/                      # [运行时] 日志
```

## Workspace 目录结构

运行时自动生成，包含训练和评测产物：

```
workspace/{task}/
├── description.md            # 任务描述（给 Agent 看）
├── instructions.txt          # Agent 使用说明
├── models/
│   ├── datasets/{task}/      # 训练数据（已删除 test 防泄漏）
│   │   └── train.jsonl
│   └── models/Qwen/xxx/      # 下载的 base model
├── output/                   # Agent 输出的训练后模型
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
├── scores.json               # 评测分数记录
├── interactions.jsonl        # HTTP 交互日志
├── baseline_score.json       # Baseline 分数
└── grading_server.log        # Grading Server 日志
```

## Grading Server API

| Endpoint | Method | 说明 |
|----------|--------|------|
| `/health` | GET | 健康检查 |
| `/submit` | POST | 提交模型评测，返回 score |
| `/best` | GET | 获取历史最高分 |
| `/history` | GET | 获取所有提交记录 |
| `/set_baseline` | POST | 设置 baseline 分数 |

### scores.json 格式

```json
{
  "submission_id": 1,
  "timestamp": "2026-01-30T12:00:00",
  "model_path": "workspace/gsm8k/output",
  "score": 0.65,
  "baseline_score": 0.45,
  "improvement": 0.20,
  "elapsed_seconds": 120.5,
  "metrics": {...}
}
```

## 添加新 Benchmark

以添加 `deep_research` 为例：

### 1. 注册任务配置

编辑 `tasks/__init__.py`，在 `TASKS` 字典中添加：

```python
"deep_research": TaskConfig(
    id="deep_research",
    type="static",                    # static | interactive | repo
    source="your/dataset",            # HuggingFace dataset 或 git URL
    subset=None,                      # dataset subset（可选）
    split="train",                    # 训练数据 split
    remove_test=True,                 # 是否删除 test split 防泄漏
    eval_type="opencompass",          # 评测类型：opencompass | 自定义
    eval_config={                     # 评测配置
        "dataset": "opencompass.configs.datasets.xxx"
    },
),
```

### 2. 添加任务描述

创建 `tasks/deep_research/description.md`，描述：
- 任务目标
- 数据格式（input/output 示例）
- 评测指标

### 3. 自定义评测（可选）

如果 OpenCompass 不支持，需要：

1. 在 `TaskConfig` 中设置 `eval_type="deep_research"`
2. 在 `benchmark/benchmark.py` 的 `run_benchmark()` 中添加分支：

```python
if task_config.eval_type == "deep_research":
    return _run_deep_research_eval(workspace_path, model_path, ...)
```

3. 实现 `_run_deep_research_eval()` 函数，返回 `{"score": float, ...}`

### 文件清单

| 必须 | 文件 | 说明 |
|:---:|------|------|
| ✅ | `tasks/__init__.py` | 注册 TaskConfig |
| ✅ | `tasks/{task}/description.md` | 任务描述 |
| ⚪ | `benchmark/benchmark.py` | 自定义评测逻辑（仅非 OpenCompass）|
