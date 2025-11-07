# LLM Fine-tuning User Guide

## Quick Start

### Basic Commands

```bash
# Run in rdagent conda environment
conda activate rdagent

# Basic usage (dataset required)
dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset HuggingFaceH4/aime_2024

# Specify both model and dataset
dotenv run -- python rdagent/app/finetune/llm/loop.py \
    --dataset HuggingFaceH4/aime_2024 \
    --model Qwen/Qwen3-1.7B
```

### Command-Line Arguments

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--dataset` | str | **Required** - HuggingFace dataset name | `HuggingFaceH4/aime_2024` |
| `--model` | str | Optional - Base model name (auto-select if not specified) | `Qwen/Qwen3-1.7B` |
| `--step_n` | int | Optional - Number of steps (unlimited by default) | `3` |
| `--loop_n` | int | Optional - Number of loops (unlimited by default) | `1` |
| `--timeout` | str | Optional - Total timeout duration | `3600` |

---

## Environment Variable Configuration

All environment variables should be configured in the `.env` file at the project root, using the `FT_` prefix.

### Core Settings

#### 1. File Path (Required)

```bash
FT_FILE_PATH=/path/to/finetune/workspace
```

**Purpose**: Specifies the root path for the fine-tuning workspace where all data, models, and outputs are stored.

**Directory Structure**:
```
FT_FILE_PATH/
├── datasets/          # Dataset storage
│   └── <dataset>/     # e.g., HuggingFaceH4/aime_2024
├── models/            # Model storage
│   └── <model>/       # e.g., Qwen/Qwen3-1.7B
├── benchmarks/        # Benchmark cache
└── output/            # Training outputs (adapters, etc.)
```

---

### Docker Configuration

#### 2. Docker Cache Control

```bash
FT_DOCKER_ENABLE_CACHE=True
```

**Purpose**: Enable Docker cache to speed up repeated runs (default: `False`)
- `True`: Keep container and image cache, suitable for development
- `False`: Clean up each time, suitable for production

#### 3. LLaMA Factory Update

```bash
FT_UPDATE_LLAMA_FACTORY=False
```

**Purpose**: Whether to update LLaMA Factory code on each run (default: `True`)
- `True`: Pull latest code from GitHub each time (slow but guaranteed latest)
- `False`: Use already downloaded version (fast, suitable for stable environments)

---

### Benchmark Evaluation Configuration

#### 4. Benchmark Dataset List

```bash
# Note: Must be valid JSON array, no trailing comments
FT_BENCHMARK_DATASETS=["aime25","gsm8k","mmlu"]
```

**Purpose**: Specify benchmark datasets for evaluation (default: `["aime25"]`)

**Supported Datasets**:
- `aime25` - AIME 2025 Math Competition (uses LLM judge scoring)
- `aime24` - AIME 2024 Math Competition
- `gsm8k` - Grade School Math 8K
- `mmlu` - Massive Multitask Language Understanding
- `math` - MATH reasoning dataset

**Mapping Rule**: The system automatically maps short names to OpenCompass full dataset names, e.g.:
- `aime25` → `aime2025_llmjudge_gen_5e9f4f`

#### 5. Benchmark Sample Limit

```bash
FT_BENCHMARK_LIMIT=50
```

**Purpose**: Limit the number of samples used per benchmark (default: `None`, full evaluation)
- Used for quick testing and debugging
- Setting to small numbers (e.g., `10`, `50`) can significantly reduce evaluation time
- Production environments should leave unset or set to `None`

**Examples**:
```bash
# Quick test: use only 10 samples per dataset
FT_BENCHMARK_LIMIT=10

# Full evaluation: use all samples
# FT_BENCHMARK_LIMIT=   # Leave empty or unset
```

#### 6. Benchmark Timeout

```bash
FT_BENCHMARK_TIMEOUT=3600
```

**Purpose**: Timeout for benchmark evaluation in seconds (default: `3600`)

---

### Judge API Configuration (for benchmarks requiring LLM scoring like AIME)

#### 7. Judge Model

```bash
FT_JUDGE_MODEL=gpt-4
```

**Purpose**: Specify the LLM model for scoring (default: `gpt-4`)

#### 8. Judge API Key

```bash
FT_JUDGE_API_KEY=sk-xxxxx
```

**Purpose**: API key for the judge model (optional, reads from environment if not set)

#### 9. Judge API Base URL

```bash
FT_JUDGE_API_BASE=http://your-api-endpoint:8000
```

**Purpose**: API base URL for the judge model (optional, uses default if not set)

---

### Training Timeout Configuration

#### 10. Debug Mode Timeout

```bash
FT_DEBUG_TIMEOUT=36000
```

**Purpose**: Training timeout in debug mode in seconds (default: `36000` = 10 hours)

#### 11. Full Training Timeout

```bash
FT_FULL_TIMEOUT=360000
```

**Purpose**: Timeout for full training in seconds (default: `360000` = 100 hours)

---

## Complete Configuration Examples

### Development/Debug Configuration

```bash
# Base path
FT_FILE_PATH=/home/user/workspace/finetune

# Docker config (enable cache to speed up debugging)
FT_DOCKER_ENABLE_CACHE=True
FT_UPDATE_LLAMA_FACTORY=False

# Benchmark config (quick testing)
FT_BENCHMARK_DATASETS=["aime25","gsm8k"]
FT_BENCHMARK_LIMIT=10

# Judge API (for AIME scoring)
FT_JUDGE_MODEL=gpt-4
FT_JUDGE_API_KEY=sk-xxxxx
FT_JUDGE_API_BASE=http://10.150.240.117:38805

# Timeouts (shorter, suitable for debugging)
FT_DEBUG_TIMEOUT=3600
FT_FULL_TIMEOUT=36000
```

### Production Configuration

```bash
# Base path
FT_FILE_PATH=/data/finetune

# Docker config (no cache, keep environment clean)
FT_DOCKER_ENABLE_CACHE=False
FT_UPDATE_LLAMA_FACTORY=True

# Benchmark config (full evaluation)
FT_BENCHMARK_DATASETS=["aime25","aime24","gsm8k","mmlu","math"]
# FT_BENCHMARK_LIMIT=   # Unset to use full datasets

# Judge API
FT_JUDGE_MODEL=gpt-4
FT_JUDGE_API_KEY=sk-xxxxx
FT_JUDGE_API_BASE=http://your-api-endpoint:8000

# Timeouts (longer, suitable for full training)
FT_DEBUG_TIMEOUT=36000
FT_FULL_TIMEOUT=360000
FT_BENCHMARK_TIMEOUT=7200
```

---

## Workflow

```
1. Start → loop.py --dataset <name> --model <name>
2. Download → Auto-download dataset and model to FT_FILE_PATH
3. Hypothesis Generation → LLM generates training hypotheses
4. Code Generation → Generate LLaMA Factory config files
5. Training → Execute training in Docker
6. Benchmark → Automatically evaluate on multiple benchmark datasets
7. Feedback → Generate feedback based on results, iterate to next round
```

---

## Code Reference

- Main entry: [rdagent/app/finetune/llm/loop.py](loop.py)
- Configuration: [rdagent/app/finetune/llm/conf.py](conf.py)
- Benchmark evaluation: [rdagent/scenarios/finetune/train/benchmark.py](../../../scenarios/finetune/train/benchmark.py)
- Runner implementation: [rdagent/scenarios/finetune/train/runner.py](../../../scenarios/finetune/train/runner.py)
