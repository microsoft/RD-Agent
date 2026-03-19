# AutoRL-Bench Task Instructions

You are an RL training agent. Your goal is to improve the model through RL post-training.

**Core objective**: maximize the score within the fixed time budget (12 hours by default). You may submit multiple times and iterate based on feedback.

## Key Information (Read First)
- **Workspace restriction**: the current directory is the workspace. Use only relative paths and do not `cd` outside it.
- **Single source of truth for time**: read `./run_meta.json` first.
- **Evaluation endpoint**: `POST $GRADING_SERVER_URL/submit`

## Environment Variables
- TASK: task name
- BASE_MODEL: base model name
- MODEL_PATH: base model path (read-only)
- DATA_PATH: training data path (read-only)
- OUTPUT_DIR: model output directory (submit a model path under this directory)
- GRADING_SERVER_URL: grading service URL

## Time and Budget Signals (Single Source of Truth)
The default budget is **12 hours (43200 seconds)**. Always trust `run_meta.json`.

Fields in `run_meta.json`:
- start_time: task start time (Unix timestamp in seconds)
- timeout_s: total time limit (seconds)
- last_submit_time: last submission time (Unix timestamp in seconds)
- end_time: task end time (Unix timestamp in seconds)

Optional API:
- GET `$GRADING_SERVER_URL/time`
  - Returns: `start_time / timeout_s / last_submit_time / end_time / now / remaining`

## Workspace and Directory Layout
**Your current directory is the workspace. All required files are located under the current directory.**
- **Do not `cd` outside the current directory**. Do not access parent directories or unrelated paths.
- **Use only relative paths** such as `./code/train.py`, not absolute paths.
- If you see a symlink pointing outside, ignore that fact and access it through the relative path here.

Directory structure:
```text
./
├── code/               # Your code area (put all self-written code here)
├── data/               # Training data (read-only)
├── models/             # Base model (read-only)
├── output/             # Model outputs (save trained models here)
├── description.md      # Task description (required reading)
├── instructions.md     # This file
├── run_meta.json       # Time and budget signals (single source of truth)
└── ...                 # Benchmark-specific files (use ls to see the full list)
```

**Run `ls` first to inspect all available files in the current directory.** Different benchmark types may provide different extra files:
- **Interactive environments** (such as ALFWorld): may provide `eval.py` (environment interaction plus evaluation logic), prompt templates, config files, and similar artifacts. These are critical references for writing training code.
- **Static datasets** (such as GSM8K): usually expose training samples mainly through files under `data/`.

**Notes**:
- `code/`: write your code here. You may organize filenames and structure freely.
- `output/`: store trained model artifacts here. You may keep multiple versions such as `output/v1/` and `output/v2/`, and specify the exact path at submission time.

## Task Loop (Improve Score Within the Budget)
1. Explore the workspace. Read `description.md`, `instructions.md`, and other relevant files. If `eval.py` is present, read it carefully.
2. Write code under `code/` and train a model. SFT, GRPO, PPO, and other methods are all allowed.
3. Save the resulting model to `$OUTPUT_DIR` such as `output/v1`.
4. Submit it for evaluation through `POST $GRADING_SERVER_URL/submit`.
5. Adjust your strategy based on the returned score and **keep iterating toward a better model within the remaining time**.

## API
```bash
# Submit a model for evaluation (returns score, improvement, and best)
curl -X POST "$GRADING_SERVER_URL/submit" \
    -H "Content-Type: application/json" \
    -d '{"model_path": "'$OUTPUT_DIR'/v1"}'

# Evaluate on a specific GPU (optional; GPU 0 is used by default)
curl -X POST "$GRADING_SERVER_URL/submit" \
    -H "Content-Type: application/json" \
    -d '{"model_path": "'$OUTPUT_DIR'/v1", "gpu": "0"}'

# Multi-GPU evaluation
curl -X POST "$GRADING_SERVER_URL/submit" \
    -H "Content-Type: application/json" \
    -d '{"model_path": "'$OUTPUT_DIR'/v1", "gpu": "2,3"}'

# Query time and budget (prefer run_meta.json; this API is supplementary)
curl "$GRADING_SERVER_URL/time"

# Health check (returns available GPU list and related status)
curl "$GRADING_SERVER_URL/health"
```

### `/submit` Parameters
| Parameter | Type | Required | Description |
|------|------|------|------|
| model_path | string | Yes | Model path |
| gpu | string | No | Requested GPU(s), such as `"0"`, `"1"`, or `"0,1"`. Must be chosen from the available GPU list. If omitted, the first available GPU is used by default. You can inspect the available list through `/health`. |

### `/submit` Response Example
```json
{
  "submission_id": 3,
  "score": 65.0,
  "baseline_score": 45.0,
  "improvement": 20.0,
  "best": {"submission_id": 2, "score": 68.0},
  "total_submissions": 3
}
```

## Important Notes
- **Do not directly submit a copied or symlinked base model**. An untrained base model only receives the baseline score (`improvement = 0`), which wastes a submission. You must train before submitting.
- You may submit multiple model versions. The system automatically tracks the best score.
- Use the remaining time carefully and iterate based on score feedback.
- **You must submit a full model**. The evaluation system does not support a LoRA adapter directory alone. If you train with LoRA/PEFT, merge before saving: `model = model.merge_and_unload(); model.save_pretrained(output_path); tokenizer.save_pretrained(output_path)`.
- After saving a model with `trl`, the `extra_special_tokens` field in `tokenizer_config.json` may be stored as a list, while vLLM/transformers expects a dict during loading. Remove that field after saving, or evaluation may fail.
