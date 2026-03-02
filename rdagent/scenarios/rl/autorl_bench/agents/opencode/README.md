# OpenCode Agent

基于 [opencode-rl](https://github.com/shatianming5/opencode-rl) 的固定阶段 Pipeline Agent，使用大模型（如 GPT-5.2）通过 OpenCode 驱动代码生成→训练→评测→反馈的迭代循环。

## 架构说明

OpenCode Agent 采用**外挂式设计**：核心 Pipeline 代码维护在独立的 [opencode-rl](https://github.com/shatianming5/opencode-rl) 仓库中，RD-Agent 通过 `start.sh` 调用外部 opencode-rl 目录。

```
RD-Agent (autorl_bench)          opencode-rl (外部独立 repo)
┌─────────────────────┐          ┌──────────────────────────┐
│ run.py              │          │ main.py                  │
│   ↓                 │  exec    │ pipeline/                │
│ start.sh ─────────────────────→│ runner_fsm/              │
│   (设置环境变量)     │          │ benchmarks/              │
│                     │          │   gsm8k, humaneval, ...  │
│ Grading Server      │◄─ HTTP ──│   smith-*, alfworld, ... │
│ (评分 & 模型管理)    │          └──────────────────────────┘
└─────────────────────┘
```

**好处**：
- opencode-rl 可以独立开发、测试、迭代，不受 RD-Agent 发版周期限制
- 支持独立运行（不依赖 RD-Agent）或作为 Agent 插件运行
- 通过 `OPENCODE_RL_ROOT` 环境变量灵活切换版本

## 快速开始

### 1. 准备 opencode-rl

```bash
# 克隆 opencode-rl 到本地（如果还没有的话）
git clone https://github.com/shatianming5/opencode-rl.git /path/to/opencode-rl
cd /path/to/opencode-rl
pip install -r requirements.txt
```

### 2. 安装 RD-Agent 依赖

```bash
cd ~/RD-Agent
pip install -e .
pip install -r rdagent/scenarios/rl/autorl_bench/requirements.txt
```

此外需要 [OpenCode](https://opencode.ai/) CLI 工具：

```bash
npm install -g opencode    # 需要 Node.js >= 18
```

### 3. 配置 `.env`

在 RD-Agent 根目录的 `.env` 中添加以下配置：

```env
# LLM API（必需）
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://your-api-endpoint/v1

# OpenCode 使用的模型（推荐 gpt-5.2，默认 gpt-5）
OPENCODE_MODEL=gpt-5.2

# opencode-rl 路径（默认 /data/userdata/v-tiansha/opencode-rl）
OPENCODE_RL_ROOT=/path/to/your/opencode-rl

# GPU 配置（可选）
CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 4. 运行

```bash
cd /path/to/RD-Agent

# GSM8K
python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent opencode --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct --timeout 41600

# ALFWorld
python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent opencode --task alfworld --model Qwen/Qwen2.5-0.5B-Instruct --timeout 41600

# 后台运行
nohup python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent opencode --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct \
    --timeout 41600 > /dev/null 2>&1 &
```

### 5. 查看日志

```bash
# Agent 实时日志
tail -f workspace/gsm8k/20260301T160000_opencode/agent.log

# 评分记录
cat workspace/gsm8k/20260301T160000_opencode/scores.json
```

---

## Pipeline 执行流程

每轮迭代包含以下固定阶段：

```
Code Gen → Training → Eval → Analysis → 下一轮
   │          │         │        │
   │          │         │        └─ Agent 总结结果，规划改进方向
   │          │         └─ 提交模型到 Grading Server 评分
   │          └─ accelerate launch train.py（GRPO 训练）
   └─ Agent 生成/修改 train.py
```

- **Code Gen**：大模型（通过 OpenCode）生成训练代码 `train.py`
- **Training**：使用 `accelerate` 执行 RL 训练（GRPO）
- **Eval**：将训练后的模型提交到 Grading Server 评测
- **Analysis**：大模型分析评测结果，决定下一轮改进方向

失败时自动重试（最多 `MAX_RETRIES` 次），支持 `--resume` 断点续跑。

---

## 配置参数

以下参数在 `config.yaml` 中配置，通过环境变量传入 opencode-rl：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_ITERATIONS` | 5 | 最大迭代轮数 |
| `MAX_RETRIES` | 20 | 各阶段失败重试次数 |
| `MAX_AGENT_STEPS` | 25 | Agent 每阶段最大步数 |
| `TRAINING_TIMEOUT` | 7200 | 训练超时（秒） |
| `STALE_TIMEOUT` | 1800 | LLM 无响应超时（秒） |
| `HTTP_TIMEOUT` | 600 | HTTP 请求超时（秒） |
| `EVAL_TIMEOUT` | 7200 | 评测请求超时（秒） |

可通过 `.env` 或命令行环境变量覆盖。

---

## 目录结构

```
agents/opencode/
├── config.yaml              # Agent 注册配置（参数、启动脚本）
├── start.sh                 # 启动脚本（设置环境变量后 exec opencode-rl）
├── README.md                # 本文档
│
└── opencode-rl/             # 内置副本（fallback，优先使用外部 OPENCODE_RL_ROOT）
    └── ...
```

外部 opencode-rl 仓库结构详见：https://github.com/shatianming5/opencode-rl

---

## 自定义 opencode-rl 路径

`start.sh` 通过 `OPENCODE_RL_ROOT` 环境变量决定使用哪个 opencode-rl：

```bash
# 默认值（在 start.sh 中）
OPENCODE_RL_ROOT="${OPENCODE_RL_ROOT:-/data/userdata/v-tiansha/opencode-rl}"
```

可以在 `.env` 中覆盖：

```env
OPENCODE_RL_ROOT=/home/user/my-opencode-rl
```

或者运行时指定：

```bash
OPENCODE_RL_ROOT=/tmp/opencode-rl-dev python -m rdagent.scenarios.rl.autorl_bench.run --agent opencode --task gsm8k
```

---

## 常见问题

### LLM 长时间 "thinking"

这是 gpt-5.2 等推理模型的正常行为（生成复杂训练代码时可能思考 1-3 分钟）。推荐使用 `OPENCODE_MODEL=gpt-5.2`。

### opencode-rl 更新

opencode-rl 独立维护，更新时只需：

```bash
cd /path/to/opencode-rl
git pull
pip install -r requirements.txt  # 如果依赖有变化
```

RD-Agent 侧无需任何修改，下次运行自动使用新版本。
