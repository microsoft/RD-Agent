# 依赖更新说明

## 概述

本文档记录了将分散在多个 Dockerfile 和 requirements.txt 中的依赖合并到统一的 `environment.yml` 的过程。

## 依赖来源

### 原始_requirements.txt 文件

| 文件路径 | 主要依赖 |
|---------|---------|
| `requirements.txt` | pydantic-settings, scikit-learn, loguru, openai, litellm, langchain, docker, streamlit, flask, mlflow, prefect 等 86 项 |
| `docs/requirements.txt` | sphinx, sphinx_rtd_theme, furo |
| `rdagent/scenarios/rl/autorl_bench/requirements.txt` | trl, accelerate, peft, datasets, torch, transformers, vllm, flask |
| `rdagent/scenarios/rl/autorl_bench/benchmarks/alpacaeval/requirements.txt` | alpaca-eval |
| `rdagent/scenarios/rl/autorl_bench/benchmarks/humaneval/requirements.txt` | humen-eval (本地安装) |
| `rdagent/scenarios/rl/autorl_bench/benchmarks/webshop/requirements.txt` | webshop, gym==0.24.0, spacy==3.7.2, flask==2.2.5 等 |
| `rdagent/scenarios/rl/autorl_bench/benchmarks/alfworld/requirements.txt` | alfworld, textworld |

### Dockerfile 中的 pip 依赖

| Dockerfile | 新增依赖 |
|-----------|---------|
| `rdagent/scenarios/rl/env/docker/miniwob/Dockerfile` | miniwob, gymnasium, selenium |
| `rdagent/scenarios/rl/env/docker/evalplus/Dockerfile` | evalplus |
| `rdagent/scenarios/rl/env/docker/base/Dockerfile` | trl==0.27.0, peft, verl==0.7.0, litellm, transformers>=4.50,<5.0 |
| `rdagent/scenarios/rl/env/docker/gsm8k/Dockerfile` | (无额外依赖) |
| `rdagent/scenarios/finetune/env/docker/opencompass/Dockerfile` | opencompass[vllm], math_verify, latex2sympy2_extended |
| `rdagent/scenarios/finetune/env/docker/llm_finetune/Dockerfile` | bitsandbytes, mixture-of-depth, litellm |
| `rdagent/scenarios/kaggle/docker/mle_bench_docker/Dockerfile` | mle-bench |
| `rdagent/scenarios/kaggle/docker/kaggle_docker/Dockerfile` | (kaggle_environment.yaml 引用，文件不存在) |
| `rdagent/scenarios/kaggle/docker/DS_docker/Dockerfile` | litellm[proxy] |
| `rdagent/scenarios/data_science/sing_docker/Dockerfile` | torch_geometric, pytorch_lightning, ogb, catboost, lightgbm==3.3.5 |
| `rdagent/scenarios/qlib/docker/Dockerfile` | catboost, xgboost, tables |

## 新增依赖列表

### LLM 训练与推理
- `verl==0.7.0` - 向量强化学习
- `vllm` - 推理加速
- `mixture-of-depth` - 模型优化
- `llama-factory` - LLM 微调框架

### 评测基准
- `opencompass` - 大模型评测框架
- `math-verify` - 数学问题评测
- `latex2sympy2-extended` - LaTeX 公式解析
- `alpaca-eval` - 对话质量评估

### RL 环境
- `miniwob` - Web 交互 RL 环境
- `ogb` - 图神经网络基准数据集

### ML 工程
- `mle-bench` - ML 工程基准

### 版本固定
- `lightgbm==3.3.5` - 与 pyqlib 兼容

## 依赖分类结构

```yaml
# === 核心基础 ===       # Pydantic, Levenshtein, fuzzywuzzy
# === LLM 与 AI ===       # OpenAI, Anthropic, LiteLLM, LangChain
# === HuggingFace 生态 === # PyTorch, Transformers, PEFT, TRL, vLLM
# === 传统 ML ===         # CatBoost, XGBoost, LightGBM, Optuna
# === 实验追踪 ===        # MLflow, WandB, TensorBoard
# === Azure AI ===        # Azure Identity, AI services
# === PyQLib ===          # PyQLib
# === 并行与分布式 ===     # Pandarallel, Prefect, Ray
# === Web 框架 ===        # Streamlit, Flask, FastAPI
# === 浏览器与爬虫 ===     # Selenium, Playwright
# === PDF 处理 ===        # PyMuPDF, PyPDF
# === 数据处理 ===        # PyArrow, FastParquet
# === 代码工具 ===        # Tree-sitter, EvalPlus
# === RL 生态 ===         # Gymnasium, Stable-Baselines3
# === WebShop ===         # WebShop + 固定版本依赖
# === Agent 框架 ===      # Pydantic-AI
# === 工具库 ===          # RandomName, Genson
# === 文档工具 ===        # Sphinx, Furo
```

## 版本兼容性说明

### PyTorch 生态
- `torch>=2.0` - 灵活版本，允许 conda/pip 自动解决依赖
- `torch_geometric` - 自动匹配 PyTorch 版本
- `pytorch-lightning` - 自动匹配

### Transformers 生态
- `transformers>=4.40,<5.0` - 避免 v5.0 破坏性变更
- `peft>=0.18.1` - 最新稳定版
- `trl>=0.27.0` - 最新稳定版

### Flask/WebShop 兼容性
- `flask==2.2.5` - WebShop 要求
- `Werkzeug==2.2.3` - 与 Flask 2.2.5 配套

## 使用方法

### 创建 conda 环境

```bash
# 从 environment.yml 创建
conda env create -f environment.yml

# 或使用 micromamba（更快）
micromamba create -f environment.yml
```

### 验证安装

```bash
# 激活环境
conda activate rdagent

# 验证关键包
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import qlib; print(f'QLib {qlib.__version__}')"
```

## 从 Docker 迁移

如果你之前使用 Docker 镜像，现在可以切换到 conda 环境：

```bash
# 1. 创建并激活环境
conda env create -f environment.yml
conda activate rdagent

# 2. 设置环境变量（如果需要）
export QLIB_DATA_PATH=~/.qlib/qlib_data/cn_data

# 3. 验证
python -c "from rdagent.utils.qlib import get_qlib_data_path; print(get_qlib_data_path())"
```

## 维护说明

1. `environment.yml` 包含所有运行时依赖
2. 添加新依赖时，请同时更新此文档
3. 尽量使用版本范围而非固定版本（除非有兼容性要求）
4. 定期使用 `conda list --export` 可以重新生成锁定文件

## 相关提交

- `c949449f` - refactor(qlib): Centralize Qlib data path configuration
- `03e413f4` - feat(config): Add Qlib data path configuration and download script
- `3a9dfb51` - refine(data): Update Qlib data download script
- `fbd404ad` - refactor: 统一 conda 环境配置并提交依赖定义
- `b40a00d7` - refactor: 补充 Dockerfile 中的遗漏依赖
