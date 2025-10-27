# OpenCompass Benchmark Integration

Automatic evaluation of fine-tuned LLMs using OpenCompass in Docker.

## Overview

After training succeeds, the system automatically evaluates your model on standard benchmarks:

```
Training → Adapter Files → Benchmark Evaluation → Results & Feedback
```

**Design Principles:**
- ✅ **Fully Automated** - No manual intervention needed
- ✅ **Docker Isolated** - Zero dependency conflicts with main environment
- ✅ **Simple Configuration** - Just list dataset names
- ✅ **Extensible** - Easy to add parameters in the future
- ✅ **No Redundancy** - Environment variables only, no JSON config files

## Quick Start

### 1. Build OpenCompass Docker Image

```bash
cd rdagent/scenarios/finetune/docker/opencompass
docker build -t rdagent-opencompass:latest .
```

### 2. Configure Datasets

Edit `rdagent/app/finetune/llm/conf.py`:

```python
benchmark_datasets: list[str] = ["mmlu", "gsm8k"]  # Your chosen benchmarks
benchmark_timeout: int = 3600  # Max execution time (seconds)
```

### 3. Run Fine-tuning Loop

```bash
python -m rdagent.app.finetune.llm.loop
```

Benchmarks run automatically after training succeeds.

## Architecture

### Files

- `benchmark.py` - `FTBenchmarkEvaluator` implementation
- `eval.py` - `FTRunnerEvaluator` (training validation)
- `runner.py` - Orchestrates evaluators
- `../docker/opencompass/Dockerfile` - Docker image definition
- `../docker/opencompass/eval_entrypoint.py` - Entrypoint script inside Docker

### Evaluation Flow

```
┌─────────────────────────────────────────────────────────────┐
│  LLMFinetuneRunner (runner.py)                              │
├─────────────────────────────────────────────────────────────┤
│  1. FTRunnerEvaluator                                       │
│     ├─ Check training exit code                             │
│     ├─ Verify adapter files exist                           │
│     └─ Return: Success/Failure (~10 sec)                    │
│                                                              │
│  2. FTBenchmarkEvaluator (only if step 1 succeeds)          │
│     ├─ Validate adapter files (safetensors/bin + config)    │
│     ├─ Generate model abbreviation (for result tracking)    │
│     ├─ Launch OpenCompass Docker with env vars              │
│     ├─ Parse results from CSV                               │
│     └─ Return: Scores + Report (minutes to hours)           │
└─────────────────────────────────────────────────────────────┘
```

### Configuration System

RDAgent passes configuration via **environment variables** (no JSON files):

```python
# benchmark.py → get_benchmark_env()
env_vars = {
    "BENCHMARK_DATASETS": "mmlu,gsm8k",           # Comma-separated list
    "ADAPTER_PATH": "/workspace/output",          # Path to adapter files
    "BASE_MODEL": "Qwen/Qwen2-1.5B-Instruct",     # HuggingFace model ID
    "MODEL_ABBR": "qwen2-1.5b-ft-abc12345",       # Unique identifier
    "MAX_OUT_LEN": "2048",                        # Max generation length
    "BATCH_SIZE": "8",                            # Inference batch size
    "NUM_GPUS": "1",                              # GPU allocation
}
```

Inside Docker, `eval_entrypoint.py` reads these variables and generates a Python config file following OpenCompass's official format:

```python
# Auto-generated config (eval_entrypoint.py)
from opencompass.models import HuggingFacewithChatTemplate

models = [dict(
    type=HuggingFacewithChatTemplate,
    abbr='qwen2-1.5b-ft-abc12345',           # For result identification
    path='Qwen/Qwen2-1.5B-Instruct',         # Base model
    peft_path='/workspace/output',           # Adapter path
    tokenizer_path='/workspace/output',      # Use adapter's tokenizer
    max_out_len=2048,
    batch_size=8,
    run_cfg=dict(num_gpus=1),                # GPU config
    model_kwargs=dict(
        device_map='auto',
        trust_remote_code=True,
    ),
    generation_kwargs=dict(do_sample=False), # Deterministic for benchmarks
)]

# Datasets (automatically imported from OpenCompass presets)
from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
datasets = mmlu_datasets + gsm8k_datasets
```

## Supported Benchmarks

OpenCompass provides 300+ datasets. Common ones included:

### 📚 Knowledge & Reasoning
- **mmlu** - Massive Multitask Language Understanding (57 subjects, multiple-choice)
- **cmmlu** - Chinese MMLU
- **bbh** - BIG-Bench Hard (challenging reasoning tasks)
- **arc** - AI2 Reasoning Challenge
- **hellaswag** - Commonsense reasoning

### 🔢 Math & Code
- **gsm8k** - Grade School Math (8K problems, generative)
- **humaneval** - Code generation (Python functions)

### 🌐 Dataset Management

OpenCompass **automatically downloads** datasets on first run:
- Downloads from HuggingFace Hub or ModelScope
- Caches locally in `~/.cache/huggingface/datasets/`
- Supports custom datasets via Python config

**No manual dataset preparation needed!**

## Configuration Options

```python
# rdagent/app/finetune/llm/conf.py

class LLMFinetunePropSetting(ExtendedBaseSettings):
    # Dataset selection
    benchmark_datasets: list[str] = ["mmlu"]
    """
    Benchmark datasets to evaluate on.
    Supported: mmlu, gsm8k, humaneval, bbh, hellaswag, cmmlu, arc, etc.
    Empty list = skip benchmarking.
    """

    # Timeout control
    benchmark_timeout: int = 3600
    """
    Benchmark evaluation timeout in seconds.
    MMLU ~30min, GSM8K ~15min, HumanEval ~10min (varies by model size and GPU).
    """
```

## OpenCompass Config Format

OpenCompass supports **three configuration methods**:

### 1. Python Config File (What we use)
```python
from opencompass.models import HuggingFacewithChatTemplate

models = [dict(
    type=HuggingFacewithChatTemplate,
    abbr='model-name',                    # Required: result identification
    path='base/model',                    # Required: base model path
    peft_path='/path/to/adapter',         # Required: adapter path
    max_out_len=2048,                     # Required: max output length
    batch_size=8,                         # Required: batch size
    run_cfg=dict(num_gpus=1),             # Required: GPU allocation
    tokenizer_path='/path/to/adapter',    # Recommended: custom tokenizer
    model_kwargs=dict(...),               # Recommended: loading params
    generation_kwargs=dict(...),          # Recommended: generation params
)]
```

### 2. CLI Shortcut (Not used in RDAgent)
```bash
opencompass --models hf_qwen2_1_5b_instruct --datasets mmlu gsm8k
```

### 3. mmengine.Config Object (Not used in RDAgent)
```python
from mmengine.config import Config
cfg = Config.fromfile('config.py')
```

**We use Method 1 (Python config)** because it's:
- Most flexible for programmatic generation
- Easiest to debug (readable Python code)
- Official recommendation for complex setups

## Docker Implementation

### Dockerfile
Located at `rdagent/scenarios/finetune/docker/opencompass/Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl vim git build-essential git-lfs unzip \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCompass and dependencies
RUN pip install opencompass peft transformers accelerate

WORKDIR /workspace
COPY eval_entrypoint.py /app/eval_entrypoint.py
```

### Entrypoint Script
Located at `rdagent/scenarios/finetune/docker/opencompass/eval_entrypoint.py`:

- Reads environment variables
- Generates OpenCompass Python config
- Invokes `opencompass.cli.main`
- Handles errors with detailed logging

## Skipping and Failure Conditions

### When Benchmark is Skipped (not an error)
```python
benchmark_datasets = []  # Empty list → skip with final_decision=True
```

### When Benchmark Fails (error)
- Training failed (FTRunnerEvaluator.final_decision=False)
- Adapter files missing or corrupt
- OpenCompass execution error (exit_code != 0)
- Results parsing failure

## Results

### Location
```
workspace/benchmark_results/
├── TIMESTAMP/
│   ├── summary/
│   │   └── summary_TIMESTAMP.csv  # Main results
│   ├── predictions/  # Model outputs
│   └── configs/      # Run configs
```

### Format
CSV with columns: `dataset`, `score`, `metric`

### Example Feedback
```
Execution: Benchmark completed: 2/2 datasets
Return Checking:
  Benchmark Results:
    mmlu: 62.3
    gsm8k: 45.7
Code: Average Score: 54.0%
Final Decision: True
```

## Troubleshooting

### 1. Docker Image Not Found
```bash
# Build the image
cd rdagent/scenarios/finetune/docker/opencompass
docker build -t rdagent-opencompass:latest .
```

### 2. Adapter Files Missing
- Check training output: `workspace/output/`
- Should contain: `adapter_model.safetensors` (or `.bin`) + `adapter_config.json`
- If missing, training likely failed

### 3. Timeout
```python
# Increase timeout in conf.py
benchmark_timeout = 7200  # 2 hours
```

### 4. Unknown Dataset
```
Warning: Unknown dataset 'my_dataset', skipping
Supported datasets: mmlu, gsm8k, humaneval, bbh, hellaswag, cmmlu, arc
```

Add to `eval_entrypoint.py`:
```python
dataset_map = {
    "my_dataset": "from opencompass.configs.datasets.my_dataset import my_datasets\ndatasets.extend(my_datasets)",
}
```

### 5. Out of Memory
- Reduce `batch_size` in `get_benchmark_env()`
- Use smaller model
- Increase GPU memory

## Future Extensions

Easy to extend by modifying environment variables in `get_benchmark_env()`:

```python
# Add new parameters
env.conf.env_vars = {
    ...
    "TEMPERATURE": "0.0",           # Generation temperature
    "TOP_P": "1.0",                 # Nucleus sampling
    "MAX_SEQ_LEN": "4096",          # Input sequence length
    "USE_VLLM": "true",             # vLLM acceleration
    "CUSTOM_PROMPT": "...",         # Custom prompt template
}
```

Then handle in `eval_entrypoint.py`.

## References

- OpenCompass: https://github.com/open-compass/opencompass
- LlamaFactory: https://github.com/hiyouga/LLaMA-Factory
- RDAgent FT Scenario: `rdagent/scenarios/finetune/`
