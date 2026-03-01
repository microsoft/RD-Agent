# OpenCode Agent

基于 [opencode-rl](opencode-rl/) 的固定阶段 Pipeline Agent，使用大模型（如 GPT-5.2）通过 OpenCode 驱动代码生成→训练→评测→反馈的迭代循环。

## 快速开始

### 1. 安装依赖

OpenCode Agent 的依赖已合并到 autorl_bench 主 requirements.txt，安装 RD-Agent 时一并安装即可：

```bash
cd ~/RD-Agent
pip install -e .
pip install -r rdagent/scenarios/rl/autorl_bench/requirements.txt
```

此外需要 [OpenCode](https://opencode.ai/) CLI 工具：

```bash
# 需要 Node.js >= 18
npm install -g opencode
```

### 2. 配置 `.env`

在 RD-Agent 根目录的 `.env` 中添加以下配置：

```env
# LLM API（必需）
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://your-api-endpoint/v1

# OpenCode 使用的模型（推荐 gpt-5.2，默认 gpt-5）
OPENCODE_MODEL=gpt-5.2

# GPU 配置（可选）
CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 3. 运行

```bash
cd /path/to/RD-Agent
conda activate cwy-rl

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

### 4. 查看日志

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
├── config.yaml              # Agent 注册配置
├── start.sh                 # 启动脚本（由 run.py 调用）
├── README.md                # 本文档
│
└── opencode-rl/             # Pipeline 核心代码
    ├── main.py              #   入口
    ├── requirements.txt     #   Python 依赖
    ├── pipeline.yml         #   Pipeline 阶段定义
    │
    ├── pipeline/            #   Pipeline 执行引擎
    │   ├── runner.py        #     状态机主循环 + checkpoint
    │   ├── phases.py        #     各阶段实现（code_gen/train/eval/analysis）
    │   ├── prompts.py       #     Agent prompt 模板
    │   ├── types.py         #     数据类型（Phase/State/Result）
    │   ├── state.py         #     checkpoint 存取
    │   └── utils.py         #     工具函数
    │
    ├── runner_fsm/          #   OpenCode 客户端 & 工具执行
    │   ├── opencode/
    │   │   ├── client.py    #     OpenCode server 通信
    │   │   ├── llm_proxy.py #     LLM 请求代理（token 统计）
    │   │   └── tool_*.py    #     工具调用解析与执行
    │   ├── core/            #     环境设置 & 执行
    │   ├── contract/        #     合约验证 & 修复
    │   ├── hints/           #     提示注入 & 评分
    │   └── utils/           #     安全 & 子进程管理
    │
    └── benchmarks/          #   Benchmark 注册（opencode-rl 内部）
        ├── registry.py      #     自动发现
        └── gsm8k/           #     GSM8K benchmark 配置
```

---

## 常见问题

### LLM 长时间 "thinking"

这是 gpt-5.2 等推理模型的正常行为（生成复杂训练代码时可能思考 1-3 分钟）。如果使用较弱的模型（如 gpt-4o-mini），思考时间更长且效果差。推荐使用 `OPENCODE_MODEL=gpt-5.2`。

### 自定义 OpenCode 路径

如果不想用 repo 内置的 opencode-rl，可以通过环境变量指向外部安装：

```bash
export OPENCODE_RL_ROOT=/path/to/your/opencode-rl
```
