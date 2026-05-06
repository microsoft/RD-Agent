# FT Job Runner

`run_ft_job.sh` launches multiple FT-Agent runs in parallel under one job directory. It is a convenience helper for multi-GPU machines; for a single run, prefer the command in `../README.md`.

## Quick Start

From the RD-Agent repository root:

```bash
# 1. Prepare the normal FT-Agent config first.
# See rdagent/app/finetune/llm/README.md for required values.
cp .env rdagent/app/finetune/llm/job/.env

# 2. Prepare task definitions.
cp rdagent/app/finetune/llm/job/tasks.json.example rdagent/app/finetune/llm/job/tasks.json

# 3. Run the job.
bash rdagent/app/finetune/llm/job/run_ft_job.sh rdagent/app/finetune/llm/job/tasks.json
```

The script opens one tmux window per task and writes logs under `log/<job_id>/`.

## Requirements

- `jq`
- `tmux`
- `conda`
- an RD-Agent conda environment, named `rdagent` by default

If your RD-Agent environment has a different name, set `CONDA_ENV_NAME` before running the script or add it to `job/.env`:

```bash
CONDA_ENV_NAME=my-rdagent-env bash rdagent/app/finetune/llm/job/run_ft_job.sh
```

The FT training/evaluation backend is still controlled by `FT_Coder_CoSTEER_env_type` in `job/.env`. In `docker` mode, the job runner skips conda readiness checks for the training and OpenCompass environments. In `conda` mode, it waits for the `llm_finetune` and `opencompass` environments after the first task initializes them.

## Task Config

`tasks.json` contains a list of independent runs:

```json
{
  "tasks": [
    {
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "benchmark": "aime25",
      "gpus": "0,1",
      "timeout": "12h"
    },
    {
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "benchmark": "chemcotbench_mol_edit",
      "gpus": "2,3",
      "benchmark_description": "Molecule Editing - perform valid SMILES edits and return JSON: {\"output\": \"<valid SMILES>\"}."
    }
  ]
}
```

| Field | Required | Default | Description |
| --- | --- | --- | --- |
| `model` | Yes | - | Hugging Face model id. |
| `benchmark` | Yes | - | Benchmark key, such as `aime25` or `chemcotbench_mol_edit`. |
| `gpus` | No | `"0"` | Value for `CUDA_VISIBLE_DEVICES`. |
| `timeout` | No | `"12h"` | Wall-clock budget passed to the FT-Agent loop. |
| `port` | No | - | Optional local API port; writes `OPENAI_API_BASE=http://localhost:<port>` for that task. |
| `benchmark_description` | No | from `scenarios.json` | Task and output-format description. |

If `benchmark_description` is omitted, the runner looks it up in `scenarios.json`.

## Monitoring

```bash
tmux attach -t rdagent
tail -f log/<job_id>/*.log
streamlit run rdagent/app/finetune/llm/ui/app.py
```

In the Streamlit UI, select the generated job folder as the log source.
