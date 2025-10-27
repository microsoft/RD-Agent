# Benchmark Evaluation with OpenCompass

## Overview

The FT scenario uses OpenCompass to evaluate fine-tuned models on standard benchmarks. Evaluation runs automatically after training as part of the runner pipeline.

```
Training (LlamaFactory) → Training Validation → Benchmark Evaluation → Feedback
```

## Quick Start

### 1. Build Docker Image

```bash
cd rdagent/scenarios/finetune/docker/opencompass
docker build -t rdagent-opencompass:latest .
```

### 2. Configure Datasets

```python
# rdagent/app/finetune/llm/conf.py
benchmark_datasets: list[str] = ["mmlu"]  # Default: MMLU only
```

### 3. Run

```bash
python rdagent/app/finetune/llm/loop.py --dataset your_dataset --model your_model
```

## Architecture

### Files

- `benchmark.py` - FTBenchmarkEvaluator implementation
- `eval.py` - FTRunnerEvaluator (training validation)
- `runner.py` - Orchestrates both evaluators
- `../docker/opencompass/` - Docker image and entrypoint

### Evaluation Pipeline

1. **Training Validation** (FTRunnerEvaluator)
   - Check exit code
   - Verify adapter files exist
   - Fast (~10 seconds)

2. **Benchmark Evaluation** (FTBenchmarkEvaluator)
   - Only runs if training validation passes
   - Runs OpenCompass in Docker
   - Evaluates on configured benchmarks
   - Returns scores

## Configuration

```python
# rdagent/app/finetune/llm/conf.py

benchmark_datasets: list[str] = ["mmlu"]
# Supported: mmlu, gsm8k, cmmlu, humaneval, bbh, hellaswag, arc, etc.

benchmark_timeout: int = 3600
# Timeout in seconds
```

## Supported Datasets

OpenCompass supports 300+ datasets. Common ones:

### Knowledge & Reasoning
- **mmlu** - Massive Multitask Language Understanding (57 subjects)
- **cmmlu** - Chinese MMLU
- **bbh** - BIG-Bench Hard
- **arc** - AI2 Reasoning Challenge

### Math
- **gsm8k** - Grade School Math 8K
- **math** - Mathematics dataset

### Code
- **humaneval** - Code generation
- **mbpp** - Python programming

### Chinese
- **cmmlu** - Chinese knowledge
- **ceval** - Chinese evaluation

## OpenCompass Dataset Management

### How Datasets are Stored

OpenCompass uses a flexible data loading system:

1. **HuggingFace/ModelScope** - Auto-download from remote
2. **Local Cache** - `./data/` directory
3. **Custom Path** - Via dataset config

```python
# opencompass/utils/datasets_info.py
DATASETS_MAPPING = {
    "opencompass/mmlu": {
        "hf_id": "opencompass/mmlu",      # HuggingFace
        "ms_id": "opencompass/mmlu",      # ModelScope (China)
        "local": "./data/mmlu/",          # Local fallback
    },
}
```

### Dataset Types

OpenCompass evaluates different question types:

#### 1. Multiple Choice (选择题)
**Examples**: MMLU, CMMLU, ARC

**Process**:
```
Question + Options → Model → "Answer: A" → Extract "A" → Compare
```

**Components**:
- `GenInferencer` - Generate text
- `AccEvaluator` - Calculate accuracy
- `first_option_postprocess` - Extract A/B/C/D

#### 2. Math/Reasoning (数学推理)
**Examples**: GSM8K, MATH

**Process**:
```
Question → Model (CoT) → "Step 1... The answer is 42" → Extract 42 → Compare
```

**Components**:
- `GenInferencer` - Generate reasoning
- `MathEvaluator` - Extract and compare numbers
- `gsm8k_postprocess` - Parse final answer

#### 3. Code Generation (代码生成)
**Examples**: HumanEval, MBPP

**Process**:
```
Problem → Model → Code → Extract → Run Tests → Pass@k
```

**Components**:
- `GenInferencer` - Generate code
- `CodeEvaluator` - Execute tests in sandbox
- `humaneval_postprocess` - Extract code blocks

#### 4. Short Answer (简答题)
**Examples**: NQ (Natural Questions)

**Process**:
```
Question → Model → Answer Text → Fuzzy Match → Score
```

**Components**:
- `GenInferencer` - Generate answer
- `EMEvaluator` - Exact match or F1 score

#### 5. LLM Judge (需要大模型评分)
**Examples**: AlpacaEval, MT-Bench

**Process**:
```
Question → Model A + Model B → GPT-4 Judge → Winner
```

**Components**:
- `LMEvaluator` - Call judge model API
- Pairwise comparison
- **Note**: Expensive, slow, requires API key

### How to Add New Datasets

Edit `docker/opencompass/eval_entrypoint.py`:

```python
dataset_map = {
    "mmlu": "from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets\ndatasets.extend(mmlu_datasets)",
    "your_dataset": "from opencompass.configs.datasets.your_dataset.your_dataset_gen import your_dataset_datasets\ndatasets.extend(your_dataset_datasets)",
}
```

Then rebuild Docker image.

## Results

### Format

Results are saved as CSV:

```csv
model,dataset,metric,score
model-name,mmlu,accuracy,45.2
model-name,gsm8k,accuracy,58.7
```

### Parsing

The evaluator parses the latest summary file:

```python
workspace/benchmark_results/{timestamp}/summary/summary_{timestamp}.csv
```

## Docker Implementation

### Environment

Following FT scenario's pattern:

```python
from rdagent.utils.env import FTDockerEnv, FTDockerConf

conf = FTDockerConf()
conf.image = "rdagent-opencompass:latest"
env = FTDockerEnv(conf=conf)

env.conf.env_vars = {
    "BENCHMARK_DATASETS": ",".join(datasets),
    "ADAPTER_PATH": str(adapter_path),
    "BASE_MODEL": base_model,
}
```

### Execution

```python
result = implementation.execute(
    env=env,
    entry="bash run_benchmark.sh",
)
```

## Troubleshooting

### Docker Image Not Found
```bash
cd rdagent/scenarios/finetune/docker/opencompass
docker build -t rdagent-opencompass:latest .
```

### Dataset Download Failed
OpenCompass auto-downloads on first run. Requires internet access.

```bash
# Use ModelScope mirror (China)
export DATASET_SOURCE=ModelScope
```

### GPU Not Available
OpenCompass can run on CPU but is slower. Docker requires nvidia-docker for GPU support.

### Timeout
Increase timeout in config:

```python
benchmark_timeout: int = 7200  # 2 hours
```

## Development Notes

### Current Implementation Status

✅ **Completed**:
- Framework architecture
- Docker setup (simplified, following FT pattern)
- Configuration system
- Result parsing

⏳ **Minimal but Functional**:
- Supports 5 common datasets (mmlu, gsm8k, cmmlu, humaneval, bbh)
- Basic error handling
- Simple result reporting

### Design Principles

1. **Follow FT Pattern** - Use `FTDockerEnv` like training does
2. **Keep It Simple** - Minimal code, straightforward logic
3. **Easy to Extend** - Add datasets by editing entrypoint mapping

## References

- [OpenCompass GitHub](https://github.com/open-compass/opencompass)
- [OpenCompass Docs](https://opencompass.readthedocs.io/)
- [Dataset List](https://opencompass.readthedocs.io/en/latest/dataset_statistics.html)
- [Training Docker](../docker/llm_finetune_docker/Dockerfile)
- [Benchmark Docker](../docker/opencompass/Dockerfile)

