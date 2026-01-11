# LLM Fine-tuning (FT) 场景运行指南

本文档介绍如何运行 RD-Agent 的 LLM Fine-tuning 场景。

## 简介

FT 场景用于自动化优化大语言模型在特定 benchmark 上的表现。系统会自动：
1. 生成数据处理和训练代码
2. 执行模型微调
3. 在目标 benchmark 上评估模型性能
4. 根据反馈迭代改进

## 支持的 Benchmark

| 类别 | Benchmark | 数据集 | 描述 |
|------|-----------|--------|------|
| Math | `aime24`, `aime25` | `deepscaler` | AIME 数学竞赛 |
| Patent | `panorama_par4pc` | `panorama-par4pc` | 专利现有技术检索 |
| Patent | `panorama_pi4pc` | `panorama-pi4pc` | 专利段落识别 |
| Patent | `panorama_noc4pc` | `panorama-noc4pc` | 专利新颖性分类 |
| Chemistry | `chemcotbench_mol_und` | `chemcot-mol_und` | 分子理解 |
| Chemistry | `chemcotbench_mol_edit` | `chemcot-mol_edit` | 分子编辑 |
| Chemistry | `chemcotbench_mol_opt` | `chemcot-mol_opt` | 分子优化 |
| Chemistry | `chemcotbench_reaction` | `chemcot-rxn` | 化学反应预测 |

> 数据集配置位于 `rdagent/scenarios/finetune/datasets/__init__.py` 的 `DATASETS` 字典中。

>运行时agent会查看所有数据集，根据target benchmark和scenario选出与之相关的。

## 环境配置

### 1. 运行环境

确保已安装 `rdagent` 主运行环境，其他需要的运行环境会自动创建

> 在 `.env` 配置文件中通过设置  `FT_Coder_CoSTEER_env_type = conda/docker` 来配置

### 2. .env 配置文件

在项目根目录创建 `.env` 文件，参考以下模板：

```bash
# ========== API Configuration ==========
BACKEND=rdagent.oai.backend.LiteLLMAPIBackend
CHAT_MODEL=gpt-5.2
CHAT_TEMPERATURE=1
CHAT_STREAM=True
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=http://your-api-endpoint

EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_USE_AZURE=True

# ========== Global Configs ==========
MAX_RETRY=12000
RETRY_WAIT_SECONDS=5
MULTI_PROC_N=16
STEP_SEMAPHORE=1

# ========== Cache Settings ==========
DUMP_CHAT_CACHE=False
USE_CHAT_CACHE=False
DUMP_EMBEDDING_CACHE=True
USE_EMBEDDING_CACHE=True
LOG_LLM_CHAT_CONTENT=True

CHAT_FREQUENCY_PENALTY=0.1
CHAT_PRESENCE_PENALTY=0.0

# ========== FT Scenario Specific ==========
FT_FILE_PATH=/path/to/your/finetune/workspace

# Environment type: docker or conda
# Set to "conda" when Docker is unavailable
FT_Coder_CoSTEER_env_type=conda

# Docker settings (only used when env_type=docker)
FT_DOCKER_ENABLE_CACHE=True
FT_UPDATE_LLAMA_FACTORY=False

# Data processing API concurrency (adjust based on target API capacity)
FT_API_MAX_WORKERS=1000

# Data processing Model
FT_STRONG_MODELS='["gpt-5", "gpt-5.1"]'
FT_WEAK_MODELS='["gpt-4o-mini"]'

# Benchmark and target (can be overridden in script)
FT_TARGET_BENCHMARK=aime25
FT_USER_TARGET_SCENARIO="I need to enhance the model's performance on math reasoning tasks."

# Timeout settings
FT_DATA_PROCESSING_TIMEOUT=28800

# Judge settings (optional)
# FT_JUDGE_MODEL=gpt-5.1
# FT_JUDGE_RETRY=10

REASONING_THINK_RM=True

# ========== Logging ==========
LOG_FORMAT_CONSOLE="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | <cyan>{process}</cyan> | {name}:{function}:{line} - {message}"

# ========== HuggingFace ==========
HF_TOKEN=hf_xxx
```

## 运行方法

### 基本命令

```bash
# 激活 conda 环境
conda activate rdagent

# 运行 FT 场景
dotenv run -- python rdagent/app/finetune/llm/loop.py --base-model <MODEL>
```

### 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--base-model` | 基础模型名称（必需，其他都可以不填） | `Qwen/Qwen2.5-7B-Instruct` |
| `--benchmark` | 目标 benchmark | `aime25` |
| `--benchmark-description` | Benchmark 描述 | - |
| `--dataset` | 指定数据集 | - |
| `--step-n` | 步数限制 | `10` |
| `--loop-n` | 循环次数限制 | `5` |
| `--timeout` | 总时间限制 | - |

### 运行示例

```bash
# 在 AIME25 上微调 Qwen2.5-7B
dotenv run -- python rdagent/app/finetune/llm/loop.py \
    --base-model Qwen/Qwen2.5-7B-Instruct

# 指定 GPU 运行
CUDA_VISIBLE_DEVICES=0,1 dotenv run -- python rdagent/app/finetune/llm/loop.py \
    --base-model Qwen/Qwen2.5-7B-Instruct

# 限制循环次数
dotenv run -- python rdagent/app/finetune/llm/loop.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --loop-n 3
```

### 多任务并行运行

创建 `tasks.json` 配置文件：
```json
{
  "tasks": [
    {"model": "Qwen/Qwen2.5-7B-Instruct", "benchmark": "aime25", "gpus": "0,1"},
    {"model": "Qwen/Qwen2.5-7B-Instruct", "benchmark": "gsm8k", "gpus": "2,3"}
  ]
}
```

使用 `run_ft_deploy.sh` 脚本运行：
```bash
./run_ft_deploy.sh tasks.json           # 正常运行
./run_ft_deploy.sh tasks.json --dry-run # 仅预览配置
./run_ft_deploy.sh tasks.json --no-sync # 禁用 blob 同步
```

<details>
<summary>run_ft_deploy.sh 脚本参考</summary>

```bash
#!/bin/bash
# 多任务并行部署脚本（简化版）

RDAGENT_DIR="$HOME/RD-Agent"
ENV_TEMPLATE=".env.ft"
STAGGER_DELAY=60

cd "$RDAGENT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rdagent

CONFIG_FILE="${1:-tasks.json}"
NUM_TASKS=$(jq '.tasks | length' "$CONFIG_FILE")

for ((i=0; i<NUM_TASKS; i++)); do
    model=$(jq -r ".tasks[$i].model" "$CONFIG_FILE")
    benchmark=$(jq -r ".tasks[$i].benchmark" "$CONFIG_FILE")
    gpus=$(jq -r ".tasks[$i].gpus" "$CONFIG_FILE")

    # 更新 .env 中的 benchmark
    cp "$ENV_TEMPLATE" .env
    sed -i "s|^FT_TARGET_BENCHMARK=.*|FT_TARGET_BENCHMARK=$benchmark|" .env

    CUDA_VISIBLE_DEVICES=$gpus \
    dotenv run -- python rdagent/app/finetune/llm/loop.py --base-model "$model" &

    # 首个任务等待环境创建，后续任务错开启动
    [[ $i -eq 0 ]] && sleep 120 || sleep $STAGGER_DELAY
done

wait
```

</details>

## Blob 日志同步

使用 Azure Blob 在多台机器间同步日志文件。

### 1. 生成 SAS Token

```bash
# 首先登录 Azure CLI
az login

# 生成 Token（默认有效期 7 天）
bash rdagent/utils/blob/gen_token.sh

# 或指定过期时间
bash rdagent/utils/blob/gen_token.sh 2025-01-31T00:00Z
```

Token 会保存到 `git_ignore_folder/.az_sas_token`。

### 2. 同步日志

同步路径：`log/` ↔ `blob://epeastus/rdagent/FinetuneAgenticLLM/FT_qizheng/logs`

```bash
# 上传本地日志到 Blob
bash rdagent/utils/blob/azsync.sh up

# 从 Blob 下载日志到本地
bash rdagent/utils/blob/azsync.sh down
```

> 如需修改远程路径，编辑 `rdagent/utils/blob/azsync.sh` 中的 `REMOTE_PATH` 变量。

## 日志查看

运行日志保存在 `log/` 目录下：

```
log/
└── 2025-01-01_12-00-00-123456/
    ├── Loop_0/
    │   ├── direct_exp_gen/   # 假设生成
    │   ├── coding/           # 代码生成
    │   ├── running/          # 训练执行
    │   └── feedback/         # 反馈总结
    └── Loop_1/
        └── ...
```


