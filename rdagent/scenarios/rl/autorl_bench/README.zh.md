# AutoRL-Bench

AutoRL-Bench 用于评测强大 LLM agent 是否能在固定工程预算下，自主驱动后训练流程，包括 RL 风格训练、多次评测和迭代优化，并真正提升一个较小模型的表现。

> 核心问题：给定一个 benchmark 及其 baseline 分数，LLM 驱动的 workflow 能否在限定预算内把目标模型提升到 baseline 之上？

| 角色 | 示例 | 职责 |
|------|------|------|
| **Benchmark** | GSM8K、HumanEval、ALFWorld、WebShop | 提供任务环境和评分逻辑 |
| **目标模型** | Qwen2.5-1.5B / 7B | 被优化的小模型 |
| **驱动模型** | GPT-5.2 等 API 模型 | 负责提出代码、奖励、超参数和训练策略 |

---

## 快速开始

### 1. 安装环境

```bash
# 1a. 克隆 RD-Agent
git clone git@github.com:microsoft/RD-Agent.git ~/RD-Agent
cd ~/RD-Agent

# 1b. AutoRL-Bench 基础环境
conda create -n autorl python=3.10 -y
conda activate autorl
pip install -e .
pip install -r rdagent/scenarios/rl/autorl_bench/requirements.txt

# 1c. 各 benchmark 额外依赖

# ALFWorld
pip install -r rdagent/scenarios/rl/autorl_bench/benchmarks/alfworld/requirements.txt

# HumanEval
git clone https://github.com/XianBW/human-eval.git ~/human-eval
cd ~/human-eval && pip install -e .
cd ~/RD-Agent

# WebShop（需要 Java 11+）
conda install -c conda-forge openjdk=11 faiss-cpu -y
pip install -r rdagent/scenarios/rl/autorl_bench/benchmarks/webshop/requirements.txt
python -c "from spacy.cli.download import download; download('en_core_web_sm')"

# AlpacaEval 2.0
pip install -r rdagent/scenarios/rl/autorl_bench/benchmarks/alpacaeval/requirements.txt

# 1d. 可选：OpenHands 运行环境
git clone git@github.com:couragec/openhands-rl.git ~/openhands-rl
conda create -n openhands python=3.12 -y
conda run -n openhands pip install -r ~/openhands-rl/requirements.txt
```

### 2. 配置 `.env`

```bash
cp .env.example .env
```

关键字段：

```env
# agent backend 使用的 API 配置
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://your-api-endpoint/v1
CHAT_MODEL=gpt-5.2

# 可选的 OpenHands 覆盖项
# CONDA_ENV_OPENHANDS=openhands
# OPENHANDS_RL_ROOT=$HOME/openhands-rl

# 可选的 rl-smith benchmark 目录
# SMITH_BENCH_DIR=/path/to/rl-smith/benchmarks
```

### 3. 运行实验

```bash
cd /path/to/RD-Agent
conda activate autorl

# 示例 GRPO agent
python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent example_agent --task gsm8k --model Qwen/Qwen2.5-1.5B --timeout 7200

# OpenHands on GSM8K
python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent openhands --task gsm8k --model Qwen/Qwen2.5-1.5B --timeout 41600

# OpenHands on ALFWorld
python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent openhands --task alfworld --model Qwen/Qwen2.5-1.5B-Instruct --timeout 41600

# 自动发现的 rl-smith benchmark
python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent openhands --task smith-bbh --model Qwen/Qwen2.5-1.5B --timeout 7200

# 后台运行
nohup python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent openhands --task alfworld --model Qwen/Qwen2.5-1.5B-Instruct \
    --timeout 41600 > /dev/null 2>&1 &
```

数据下载是自动完成的。首次运行某个 benchmark 时，会调用对应的 `data.py` 准备训练数据。

- GSM8K：从 HuggingFace 下载
- HumanEval：从 HuggingFace 下载
- ALFWorld：通过 `alfworld-download` 下载

### 4. 查看结果

```bash
# 查看运行日志
tail -f workspace/alfworld/20260228T100000_openhands/agent.log

# 查看单次 run 的提交记录
cat workspace/alfworld/20260228T100000_openhands/scores.json

# 查看全局实验日志
cat rdagent/scenarios/rl/autorl_bench/results.csv

# Streamlit 面板
streamlit run rdagent/scenarios/rl/autorl_bench/core/ui.py --server.port 8511
```

### 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--agent` | Agent 后端 | `example_agent`, `rdagent`, `openhands` |
| `--task` | Benchmark id | `gsm8k`, `alfworld`, `smith-bbh` |
| `--model` | HuggingFace 模型 repo id | `Qwen/Qwen2.5-1.5B` |
| `--timeout` | 最大运行时长（秒） | `41600` |
| `--port` | Grading server 端口 | `5000` |

---

## 核心流程

```text
run.py
  1. 准备资源：下载模型权重和 benchmark 训练数据
  2. 构建隔离 workspace，并为模型/数据建立软链接
  3. 挂载任务文件，如 description.md 和 instructions.md
  4. 启动基于 Flask 的 grading server
  5. 先评测一次 baseline，并缓存结果
  6. 在 workspace 内运行选定 agent
  7. 记录 grading server 返回的所有提交
  8. 将本次 run 摘要追加到 results.csv
```

### 资源布局

下载资源默认存放在 `git_ignore_folder/rl_files/` 下，可通过 `AUTORL_FILE_PATH` 覆盖。

```text
git_ignore_folder/rl_files/
├── models/Qwen/Qwen2.5-1.5B/
├── datasets/
│   ├── gsm8k/train.jsonl
│   └── alfworld/train -> ...
└── baseline_workspace/
    └── gsm8k_Qwen_Qwen2.5-1.5B.json
```

### Workspace 布局

每个 run 都有独立目录，位于 `workspace/<task>/<run_id>_<agent>/`。

```text
workspace/gsm8k/
├── 20260211T143000_openhands/
│   ├── code/
│   │   ├── train.py
│   │   └── ...
│   ├── output/
│   │   ├── v1/
│   │   └── v2/
│   ├── models/Qwen/Qwen2.5-1.5B -> rl_files/models/...
│   ├── data -> rl_files/datasets/gsm8k/
│   ├── description.md -> benchmarks/gsm8k/description.md
│   ├── instructions.md -> core/instructions.md
│   ├── scores.json
│   └── grading_server.log
└── 20260211T160000_rdagent/
```

每次 `run.py` 调用都是一个独立评测单元。Agent 可以在时限内多次训练、多次提交，最终分数仅取该次 run 内的最佳提交。

### `results.csv`

`autorl_bench/results.csv` 是运行日志，用于后续分析和论文制表，不参与评分逻辑。

```csv
run_id,timestamp,task,agent,base_model,baseline,best_score,improvement,submissions,duration_s,success,workspace
20260211T143000,2026-02-11 14:30:00,gsm8k,openhands,Qwen/Qwen2.5-1.5B,21.61,22.37,0.76,3,3600,True,workspace/gsm8k/...
20260211T160000,2026-02-11 16:00:00,gsm8k,rdagent,Qwen/Qwen2.5-1.5B,21.61,23.12,1.51,7,3600,True,workspace/gsm8k/...
```

---

## Agent 环境变量

在 agent `start.sh` 中可直接读取以下变量：

| 变量 | 说明 | 示例 |
|------|------|------|
| `TASK` | 任务 id | `gsm8k` |
| `BASE_MODEL` | 模型名称 | `Qwen/Qwen2.5-1.5B` |
| `WORKSPACE` | Workspace 根目录 | `workspace/gsm8k/20260211T143000` |
| `MODEL_PATH` | 只读模型路径 | `$WORKSPACE/models/Qwen/Qwen2.5-1.5B` |
| `DATA_PATH` | 只读数据路径 | `$WORKSPACE/data` |
| `OUTPUT_DIR` | 可写输出目录 | `$WORKSPACE/output` |
| `GRADING_SERVER_URL` | Grading server 地址 | `http://localhost:5000` |

### Grading Server API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/submit` | POST | 提交 `{"model_path": "..."}`，返回 score / best / improvement |
| `/set_baseline` | POST | 设置 `{"score": 21.91}` 作为 baseline |
| `/health` | GET | 健康检查 |

`/submit` 返回示例：

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

## 仓库结构

```text
autorl_bench/
├── run.py
├── conf.py
├── core/
│   ├── evaluator.py
│   ├── opencompass.py
│   ├── server.py
│   ├── utils.py
│   └── instructions.md
├── benchmarks/
│   ├── __init__.py
│   ├── smith/
│   │   ├── __init__.py
│   │   └── per_sample_eval.py
│   ├── gsm8k/
│   │   ├── data.py
│   │   └── description.md
│   └── alfworld/
│       ├── data.py
│       ├── eval.py
│       ├── requirements.txt
│       ├── description.md
│       └── react_prompts.json
├── agents/
│   ├── registry.py
│   ├── example_agent/
│   ├── openhands/
│   └── rdagent/
└── workspace/
```

---

## 扩展 AutoRL-Bench

### 通过 `rl-smith` 添加 benchmark

将 benchmark 放在 `rl-smith/benchmarks/<name>/` 下。RD-Agent 启动时会自动发现，并注册为 `smith-<name>`。

```bash
cd /path/to/rl-smith
python generate_benchmark.py https://github.com/suzgunmirac/BIG-Bench-Hard --name my_bbh

cd /path/to/RD-Agent
python -m rdagent.scenarios.rl.autorl_bench.run \
    --task smith-my_bbh --agent openhands --model Qwen/Qwen2.5-1.5B
```

手工创建 `rl-smith/benchmarks/<name>/` 时，建议包含：

| 文件 | 必需 | 用途 |
|------|------|------|
| `config.yaml` | 是 | `name`、`eval_mode`、`expose_files` |
| `eval.py` | 是 | 导出 `evaluate(...) -> float` |
| `data/train.jsonl` | 是 | 训练数据 |
| `download_data.py` | 否 | 幂等下载脚本 |
| `description.md` | 否 | 挂载到 workspace 的说明文件 |

### 在 RD-Agent 内手工注册 benchmark

新建 `benchmarks/new_task/`，至少包含：

1. `data.py`：准备训练数据
2. `description.md`：供 agent 阅读的任务说明
3. `benchmarks/__init__.py` 中的注册项

示例：

```python
BENCHMARKS["new_task"] = BenchmarkConfig(
    id="new_task",
    evaluator_class="rdagent.scenarios.rl.autorl_bench.core.opencompass.OpenCompassEvaluator",
    data_module="rdagent.scenarios.rl.autorl_bench.benchmarks.new_task.data",
    description="New task description",
    eval_config={"dataset": "opencompass.configs.datasets.xxx"},
)
```

如果需要自定义 evaluator，再增加 `eval.py` 并实现 `BaseEvaluator` 子类即可。

### 添加新 agent

```yaml
# agents/my_agent/config.yaml
name: "My Agent"
start: "start.sh"
env_vars:
  MY_PARAM: "value"
```

```bash
#!/bin/bash
python $WORKSPACE/code/train.py --model $MODEL_PATH --data $DATA_PATH --output $OUTPUT_DIR/v1
curl -X POST $GRADING_SERVER_URL/submit \
    -H "Content-Type: application/json" \
    -d '{"model_path": "'$OUTPUT_DIR'/v1"}'
```

Agent 注册是配置驱动的。新增 `config.yaml` 即可，不需要额外修改 registry 代码。
