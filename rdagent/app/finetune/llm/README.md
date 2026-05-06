# FT-Agent: Autonomous LLM Fine-Tuning

This directory contains the RD-Agent LLM fine-tuning scenario used by **FT-Agent** in the ICML 2026 paper [FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents](https://arxiv.org/abs/2603.01712).

FT-Agent automates a benchmark-driven fine-tuning loop:

1. inspect the target benchmark and available raw datasets;
2. generate data processing code and LLaMA-Factory training configs;
3. run fail-fast validation before full training;
4. fine-tune the target model;
5. evaluate with OpenCompass and use feedback for the next iteration.

The implementation is research-oriented. A full run can download large datasets, build training/evaluation environments, call LLM APIs for data processing, and consume GPU hours.

## Supported Benchmarks

The scenario currently includes benchmark adapters for the FT-Dojo tasks and several related extensions.

| Domain | Benchmarks | Main raw dataset |
| --- | --- | --- |
| Math | `aime24`, `aime25` | `deepscaler` |
| Patent | `panorama_par4pc`, `panorama_pi4pc`, `panorama_noc4pc` | `panorama` |
| Chemistry | `chemcotbench_mol_und`, `chemcotbench_mol_edit`, `chemcotbench_mol_opt`, `chemcotbench_reaction` | `chemcot` |
| Table QA | `tablebench_data_analysis`, `tablebench_fact_checking`, `tablebench_numerical_reasoning`, `tablebench_visualization` | `tableinstruct` |
| Finance | `FinanceIQ_gen` | `financeiq` |
| Biology | `bioprobench_gen`, `bioprobench_ord`, `bioprobench_err`, `bioprobench_pqa` | `bioprobench` |

Dataset registration lives in `rdagent/scenarios/finetune/datasets/__init__.py`. Benchmark adapters live in `rdagent/scenarios/finetune/benchmark/data/adaptor.py`.

Current dataset behavior: scenario startup prepares the registered dataset resources under `FT_FILE_PATH`. The first run can therefore be large and slow. Reusing the same `FT_FILE_PATH` lets later runs reuse already downloaded assets. The `--dataset` option constrains the agent's selected dataset after preparation; it does not change the current preparation step.

## Prerequisites

- Linux.
- Docker available to the current user without `sudo`, or a compatible conda setup.
- NVIDIA GPU for realistic fine-tuning runs.
- LLM API access for the RD-Agent planner and optional data processing calls.
- Hugging Face access for target models and datasets. Some datasets may require accepting upstream licenses or setting `HF_TOKEN`.

Install RD-Agent from source when using this scenario:

```bash
git clone https://github.com/microsoft/RD-Agent
cd RD-Agent
make dev
```

## Minimal `.env`

Create `.env` in the repository root. Adjust model names and API settings to your provider.

```bash
# LLM backend
BACKEND=rdagent.oai.backend.LiteLLMAPIBackend
CHAT_MODEL=gpt-4o
CHAT_TEMPERATURE=1
CHAT_STREAM=True
OPENAI_API_KEY=<your_api_key>
# OPENAI_API_BASE=<your_api_base_if_needed>

# Embedding model used by RD-Agent infrastructure
EMBEDDING_MODEL=text-embedding-3-small

# Fine-tuning workspace. Keep this stable to reuse downloaded models/datasets.
FT_FILE_PATH=/absolute/path/to/finetune_files

# Runtime environment: docker is the default path; conda is available for local setups.
FT_Coder_CoSTEER_env_type=docker

# Target task. You may also pass these through CLI arguments.
FT_TARGET_BENCHMARK=aime25
FT_BENCHMARK_DESCRIPTION="AIME 2025 math competition problems. Each answer is an integer from 0 to 999. Expected Output Format: put the final answer within \\boxed{}, for example \\boxed{42}."

# Target model and data-processing settings
FT_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
FT_UPPER_DATA_SIZE_LIMIT=2000
FT_API_MAX_WORKERS=8
FT_STRONG_MODELS='["gpt-4o"]'
FT_WEAK_MODELS='["gpt-4o-mini"]'

# Hugging Face token, if required by a model or dataset.
# HF_TOKEN=<your_hf_token>
```

`FT_API_MAX_WORKERS` defaults to `8` to avoid surprising rate-limit and cost spikes for public users. If you have a high-throughput internal endpoint, increase it explicitly in `.env`.

## Single-Task Run

Using the CLI entry point:

```bash
rdagent llm_finetune \
  --benchmark aime25 \
  --benchmark-description "AIME 2025 math competition problems. Each answer is an integer from 0 to 999. Expected Output Format: put the final answer within \\boxed{}, for example \\boxed{42}." \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --loop-n 3 \
  --timeout 12h
```

Equivalent direct Python entry point:

```bash
dotenv run -- python rdagent/app/finetune/llm/loop.py \
  --benchmark aime25 \
  --benchmark-description "AIME 2025 math competition problems. Each answer is an integer from 0 to 999. Expected Output Format: put the final answer within \\boxed{}, for example \\boxed{42}." \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --loop-n 3 \
  --timeout 12h
```

Useful arguments:

| Argument | Meaning |
| --- | --- |
| `--base-model` | Hugging Face model id to fine-tune. Required unless set by `FT_BASE_MODEL`. |
| `--benchmark` | Target benchmark key, such as `aime25` or `chemcotbench_mol_edit`. |
| `--benchmark-description` | Natural-language task and output-format description. Required unless set in `.env`. |
| `--dataset` | Dataset name to select for the agent after dataset preparation. |
| `--upper-data-size-limit` | Maximum number of training examples used by one experiment. |
| `--loop-n` | Maximum number of RD-Agent loops. |
| `--timeout` | Overall wall-clock budget, such as `12h`. |

## Batch Runs

For multiple benchmark/model runs, use the job helper in this directory:

```bash
cp rdagent/app/finetune/llm/job/tasks.json.example rdagent/app/finetune/llm/job/tasks.json
cp .env rdagent/app/finetune/llm/job/.env
bash rdagent/app/finetune/llm/job/run_ft_job.sh rdagent/app/finetune/llm/job/tasks.json
```

The job runner reads benchmark descriptions from `rdagent/app/finetune/llm/job/scenarios.json` when a task does not provide `benchmark_description` directly.

## Logs and UI

Runs write RD-Agent traces under the configured log directory. For the Streamlit FT UI:

```bash
streamlit run rdagent/app/finetune/llm/ui/app.py
```

For the generic RD-Agent log UI:

```bash
rdagent ui --port 19899 --log-dir <your_log_folder>
```

## Notes

- The first run is expected to be slow because it may download models, datasets, LLaMA-Factory assets, and OpenCompass assets.
- Docker mode mounts models and datasets from `FT_FILE_PATH` into the training/evaluation environments.
- Generated training data must use Alpaca-style `instruction`, `input`, and `output` fields; the validator checks this before full training.
- Evaluation uses validation feedback for agent iteration and keeps the test split for final reporting/front-end display.
