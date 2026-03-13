# FT Job Runner

批量并行运行多个 LLM 微调任务的脚本。

## 快速开始

```bash
# 1. 准备环境配置
cp .env.template .env
# 编辑 .env，填入 API key 等配置

# 2. 准备任务配置
cp tasks.json.example tasks.json
# 编辑 tasks.json，定义要运行的任务

# 3. 运行
./run_ft_job.sh
```

## 用法

```bash
./run_ft_job.sh [tasks.json]
```

| 参数 | 说明 |
|------|------|
| `tasks.json` | 任务配置文件路径（可选，默认使用同目录下的 `tasks.json`） |
| `-h, --help` | 显示帮助信息 |

### 示例

```bash
# 使用默认配置
./run_ft_job.sh

# 指定自定义配置文件
./run_ft_job.sh /path/to/my_tasks.json
```

## 配置文件

### tasks.json

定义要并行运行的任务列表：

```json
{
  "tasks": [
    {
      "model": "Qwen/Qwen3-8B",
      "benchmark": "aime25",
      "gpus": "0,1"
    },
    {
      "model": "Qwen/Qwen3-8B",
      "benchmark": "gsm8k",
      "gpus": "2,3",
      "scenario": "自定义优化目标"
    }
  ]
}
```

| 字段 | 必填 | 默认值 | 说明 |
|------|:----:|--------|------|
| `model` | ✅ | - | HuggingFace 模型路径 |
| `benchmark` | ✅ | - | 评估基准（如 `aime25`, `gsm8k`） |
| `gpus` | ❌ | `"0"` | 使用的 GPU 编号 |
| `scenario` | ❌ | `"Improve model performance on {benchmark}"` | 优化目标描述 |

### .env

环境配置文件，包含 API 密钥、模型设置等。从 `.env.template` 复制并修改：

```bash
cp .env.template .env
```

主要配置项：

| 配置 | 说明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API 密钥 |
| `OPENAI_API_BASE` | API 地址 |
| `FT_Coder_CoSTEER_env_type` | 环境类型：`docker` 或 `conda` |
| `HF_TOKEN` | HuggingFace Token |

## 输出

运行后会在 `log/` 目录下创建 job 文件夹：

```
log/2025-12-23/
├── aime25_Qwen3-8B.log      # 任务日志
├── gsm8k_Qwen3-8B.log
└── aime25_Qwen3-8B/         # 任务 trace（Loop 数据）
    ├── Loop_0/
    └── ...
```

## 监控

### 命令行

```bash
# 查看所有任务日志
tail -f log/2025-12-23/*.log

# 查看特定任务
tail -f log/2025-12-23/aime25_Qwen3-8B.log
```

### Web UI

```bash
streamlit run rdagent/app/finetune/llm/ui/app.py
```

在 UI 中选择 Job Folder 为对应的日志目录即可查看运行状态。

## 依赖

- `jq`：JSON 解析工具
- `conda` 环境：`rdagent`

## 注意事项

1. 任务启动间隔默认为 60 秒（`STAGGER_DELAY`），避免同时启动造成资源竞争
2. 确保指定的 GPU 编号不冲突
3. 如果同一天多次运行，会自动创建 `log/2025-12-23_1/`、`log/2025-12-23_2/` 等目录
