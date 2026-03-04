# AutoRL-Bench

让大模型（如 GPT-5.2）自主驱动 RL 训练流程，提升小模型（如 Qwen2.5-7B）在各类 Benchmark 上的表现，并评测"大模型驱动 RL"的增益效果。

> 核心问题：给定一个 Benchmark 及其 baseline，大模型通过 Workflow 对小模型进行 RL 训练后，小模型的分数能否超过 baseline？

| 角色 | 实例 | 职责 |
|------|------|------|
| **Benchmark** | GSM8K、HumanEval、ALFWorld 等 | 提供任务环境、自动评分 |
| **小模型** | Qwen2.5-0.5B/7B | 被 RL 训练的 Agent |
| **大模型** | GPT-5.2 等 | 离线驱动 RL 优化（生成 reward、调超参等） |

---

## 快速开始

### 1. 环境安装

```bash
# --- 1a. Clone 代码 ---
git clone git@github.com:microsoft/RD-Agent.git ~/RD-Agent
cd ~/RD-Agent

# --- 1b. 基础 conda 环境 ---
conda create -n autorl python=3.10 -y
conda activate autorl
pip install -e .

# 全局依赖（trl, vllm, torch, opencompass 等）
pip install -r rdagent/scenarios/rl/autorl_bench/requirements.txt

# --- 1c. 按需安装 benchmark 额外依赖 ---

# ALFWorld
pip install -r rdagent/scenarios/rl/autorl_bench/benchmarks/alfworld/requirements.txt

# GSM8K：无额外依赖

# HumanEval
git clone https://github.com/XianBW/human-eval.git ~/human-eval
cd ~/human-eval && pip install -e .
cd ~/RD-Agent

# WebShop（需要 Java 11+）
conda install -c conda-forge openjdk=11 faiss-cpu -y
pip install -r rdagent/scenarios/rl/autorl_bench/benchmarks/webshop/requirements.txt
python -c "from spacy.cli.download import download; download('en_core_web_sm')"

# --- 1d. OpenHands Agent（如需使用）---
git clone git@github.com:couragec/openhands-rl.git ~/openhands-rl
# OpenHands 需要独立 conda 环境（Python 3.12）
conda create -n openhands python=3.12 -y
conda run -n openhands pip install -r ~/openhands-rl/requirements.txt
```

### 2. 配置 `.env`

```bash
cp .env.example .env  # 或手动创建
```

`.env` 中需要配置的关键项：

```env
# LLM API（OpenHands Agent 必需）
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://your-api-endpoint/v1
CHAT_MODEL=gpt-5.2

# OpenHands 环境（可选，有默认值）
# CONDA_ENV_OPENHANDS=openhands      # 默认 openhands
# OPENHANDS_RL_ROOT=$HOME/openhands-rl  # 默认 ~/openhands-rl
```

### 3. 运行

```bash
cd /path/to/RD-Agent
conda activate autorl

# Example Agent（简单 GRPO 训练，验证流程）
python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent example_agent --task gsm8k --model Qwen/Qwen2.5-0.5B --timeout 7200

# OpenHands Agent + GSM8K
python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent openhands --task gsm8k --model Qwen/Qwen2.5-0.5B --timeout 41600

# OpenHands Agent + ALFWorld（首次运行自动下载 ~2GB 游戏数据）
python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent openhands --task alfworld --model Qwen/Qwen2.5-0.5B-Instruct --timeout 41600

# 后台运行（推荐）
nohup python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent openhands --task alfworld --model Qwen/Qwen2.5-0.5B-Instruct \
    --timeout 41600 > /dev/null 2>&1 &
```

> **数据自动下载**：首次运行某个 benchmark 时，`run.py` 会自动调用对应 `data.py` 下载训练数据，无需手动操作。
> - GSM8K：从 HuggingFace 下载 (~5MB)
> - HumanEval：从 HuggingFace 下载 (~164 条样本)
> - ALFWorld：调用 `alfworld-download` 从 GitHub Releases 下载 (~2GB，含 json/pddl/tw-pddl/logic)

### 4. 查看结果

```bash
# 实时查看运行日志
tail -f workspace/alfworld/20260228T100000_openhands/agent.log

# 查看评分记录
cat workspace/alfworld/20260228T100000_openhands/scores.json

# 查看全局实验汇总
cat rdagent/scenarios/rl/autorl_bench/results.csv

# Web UI（Streamlit 面板）
streamlit run rdagent/scenarios/rl/autorl_bench/core/ui.py --server.port 8511
```

### 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--agent` | Agent 类型 | `example_agent`、`rdagent`、`openhands` |
| `--task` | Benchmark 任务名（对应 `benchmarks/` 子目录） | `gsm8k`、`humaneval`、`alfworld` |
| `--model` | HuggingFace 模型 repo_id，首次自动下载 | `Qwen/Qwen2.5-0.5B` |
| `--timeout` | Agent 最大运行时长（秒） | `41600`（~11.5h） |
| `--port` | Grading Server 端口（默认 5000） | `5000` |

---

## 核心流程

```
run.py 启动
 │
 ├─ 1. 准备资源：下载模型（HuggingFace）+ 下载训练数据（各 benchmark 的 data.py）
 ├─ 2. 构建 workspace：创建隔离目录、软链接模型和数据
 ├─ 3. 挂载文件：description.md + instructions.md + benchmark 特有文件
 ├─ 4. 启动 Grading Server（Flask 评测服务）
 ├─ 5. 评测 baseline：用原始模型跑一次基准分（有缓存）
 ├─ 6. 运行 Agent：Agent 在 workspace 内训练 + 多次提交评测
 ├─ 7. 收集结果：从 Grading Server 获取所有提交记录
 └─ 8. 保存结果：追加到 results.json，更新全局 best
```

### 资源存储

模型和数据下载后统一存储在 `git_ignore_folder/rl_files/`（可通过 `AUTORL_FILE_PATH` 覆盖）：

```
git_ignore_folder/rl_files/
├── models/Qwen/Qwen2.5-0.5B/    # 模型权重（snapshot_download）
├── datasets/
│   ├── gsm8k/train.jsonl         # 训练数据（agent 可见）
│   └── alfworld/train → ...      # 训练游戏数据（agent 可见，评估数据不在这）
└── baseline_workspace/           # baseline 分数缓存
    └── gsm8k_Qwen_Qwen2.5-0.5B.json
```

### Workspace（每次运行隔离）

每次运行创建独立的 workspace 目录（`workspace/<task>/<run_id>/`），通过软链接挂载资源：

```
workspace/gsm8k/
├── 20260211T143000_openhands/        # 一次独立实验（agent 在时限内的完整生命周期）
│   ├── code/                         # Agent 代码区（所有自行编写的代码）
│   │   ├── train.py                  # 训练脚本
│   │   └── ...                       # 分析、处理等其他脚本
│   ├── output/                       # 模型输出（$OUTPUT_DIR）
│   │   ├── v1/                       # 第一版模型
│   │   └── v2/                       # 第二版模型（迭代优化）
│   ├── models/Qwen/Qwen2.5-0.5B →   # 软链接 → rl_files/models/...（只读）
│   ├── data →                        # 软链接 → rl_files/datasets/gsm8k/（只读）
│   ├── description.md →              # 软链接 → benchmarks/gsm8k/description.md
│   ├── instructions.md →             # 软链接 → core/instructions.md
│   ├── scores.json                   # 本次实验内所有提交的评分记录
│   └── grading_server.log            # Grading Server 日志
└── 20260211T160000_rdagent/          # 另一次独立实验
    └── ...
```

> **评测原则**：每次实验（一次 `run.py` 调用）是一个独立的评测单元。
> Agent 在 `--timeout` 时限内可以多次训练、多次提交，最终取**本次实验内**的最高分。
> 不同实验之间完全隔离，不存在跨实验的"全局最优"。

### results.csv（实验日志）

`autorl_bench/results.csv` 是纯日志记录，用于论文实验汇总，**不参与评测逻辑**：

```csv
run_id,timestamp,task,agent,base_model,baseline,best_score,improvement,submissions,duration_s,success,workspace
20260211T143000,2026-02-11 14:30:00,gsm8k,openhands,Qwen/Qwen2.5-0.5B,21.61,22.37,0.76,3,3600,True,workspace/gsm8k/...
20260211T160000,2026-02-11 16:00:00,gsm8k,rdagent,Qwen/Qwen2.5-0.5B,21.61,23.12,1.51,7,3600,True,workspace/gsm8k/...
```

每行记录一次独立实验的结果，方便对比不同 agent 在相同条件下的表现。

---

## Agent 环境变量

Agent 启动时（`start.sh`）可用的环境变量：

| 变量 | 说明 | 示例 |
|------|------|------|
| `TASK` | 任务名 | `gsm8k` |
| `BASE_MODEL` | 模型名 | `Qwen/Qwen2.5-0.5B` |
| `WORKSPACE` | 工作根目录 | `workspace/gsm8k/20260211T143000` |
| `MODEL_PATH` | 模型路径（只读） | `$WORKSPACE/models/Qwen/Qwen2.5-0.5B` |
| `DATA_PATH` | 数据路径（只读） | `$WORKSPACE/data` |
| `OUTPUT_DIR` | 输出目录 | `$WORKSPACE/output` |
| `GRADING_SERVER_URL` | 评测服务地址 | `http://localhost:5000` |

### Grading Server API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/submit` | POST | `{"model_path": "..."}` → 返回 score + best + improvement |
| `/set_baseline` | POST | `{"score": 21.91}` → 设置 baseline |
| `/health` | GET | 健康检查 |

`/submit` 响应：

```json
{
  "submission_id": 3,
  "score": 65.0,
  "baseline_score": 45.0,
  "improvement": 20.0,
  "best": {"submission_id": 2, "score": 68.0},
  "total_submissions": 3
}
```

---

## 代码结构

```
autorl_bench/
├── run.py                    # 入口脚本
├── conf.py                   # 路径配置
│
├── core/                     # 【主干代码】
│   ├── evaluator.py          # BaseEvaluator 基类
│   ├── opencompass.py        # OpenCompassEvaluator（通用评测器）
│   ├── server.py             # Grading Server（Flask）
│   ├── utils.py              # 工具函数（下载、软链接、baseline）
│   └── instructions.md       # Agent 通用指导说明
│
├── benchmarks/               # 【Benchmark 扩展】
│   ├── __init__.py           # 注册表 BENCHMARKS
│   ├── gsm8k/
│   │   ├── data.py           # 数据下载（train split）
│   │   └── description.md
│   └── alfworld/
│       ├── data.py           # 数据下载（训练游戏数据）
│       ├── eval.py           # 自定义评测器
│       ├── requirements.txt  # 额外依赖（alfworld, textworld）
│       ├── description.md
│       └── react_prompts.json
│
├── agents/                   # 【Agent 扩展】
│   ├── registry.py           # 注册表（读 config.yaml）
│   ├── example_agent/        # 简单 GRPO 训练
│   ├── openhands/            # OpenHands SDK
│   └── rdagent/              # RD-Agent
│
└── workspace/                # [运行时] 工作区 + 结果
```

---

## 扩展指南

### 添加新 Benchmark

新建 `benchmarks/new_task/` 目录，需要 3 个文件：

**1. `data.py` — 数据下载（只给 agent 训练数据，评估数据自己管）**

```python
# benchmarks/new_task/data.py
from pathlib import Path
from loguru import logger

def download_train_data(target_dir: Path) -> None:
    """下载训练数据到 target_dir，agent 只能看到这里的内容"""
    # target_dir 会被软链接到 workspace/data
    ...
```

**2. `description.md` — 任务描述（agent 可见）**

**3. 注册到 `benchmarks/__init__.py`**

```python
BENCHMARKS["new_task"] = BenchmarkConfig(
    id="new_task",
    evaluator_class="rdagent.scenarios.rl.autorl_bench.core.opencompass.OpenCompassEvaluator",
    data_module="rdagent.scenarios.rl.autorl_bench.benchmarks.new_task.data",
    description="新任务描述",
    eval_config={"dataset": "opencompass.configs.datasets.xxx"},
)
```

如果需要自定义评测逻辑（不用 OpenCompass），再加一个 `eval.py`：

```python
# benchmarks/new_task/eval.py
from rdagent.scenarios.rl.autorl_bench.core import BaseEvaluator

class NewTaskEvaluator(BaseEvaluator):
    def __init__(self, config):
        self.config = config

    def run_eval(self, model_path: str, workspace_path: str, **kwargs) -> dict:
        return {"score": 85.0, "accuracy_summary": {...}}
```

### 添加新 Agent

```yaml
# agents/my_agent/config.yaml
name: "My Agent"
start: "start.sh"
env_vars:
  MY_PARAM: "value"
```

```bash
# agents/my_agent/start.sh
#!/bin/bash
# 在 code/ 下编写训练脚本，模型输出到 output/
python $WORKSPACE/code/train.py --model $MODEL_PATH --data $DATA_PATH --output $OUTPUT_DIR/v1
curl -X POST $GRADING_SERVER_URL/submit \
    -H "Content-Type: application/json" \
    -d '{"model_path": "'$OUTPUT_DIR'/v1"}'
```

Agent 通过 `config.yaml` 自动注册，无需修改代码。
