#AutoRL-Bench task description

You are a reinforcement learning training agent, and your goal is to improve model performance through RL Post-Training.

**Core Goal**: Improve your score as much as possible within a fixed period of time (default 12 hours), submit multiple times and iterate with feedback.

## Key information (read first)
- **Workspace restrictions**: The current directory is the workspace, only relative paths are used, and `cd` to the outside is prohibited.
- **Time single source of truth**: read `./run_meta.json` first
- **Evaluation Entrance**: `POST $GRADING_SERVER_URL/submit`

## Environment variables
- TASK: task name
- BASE_MODEL: base model name
- MODEL_PATH: base model path (read-only)
- DATA_PATH: training data path (read-only)
- OUTPUT_DIR: model output directory (specify the model path in this directory when submitting for evaluation)
- GRADING_SERVER_URL: Evaluation service address

## Time and budget signals (single source of truth)
The default budget is **12 hours (43200 seconds)**, based on `run_meta.json`.

`run_meta.json` fields:
- start_time: task start time (unix timestamp, seconds)
- timeout_s: total time limit (seconds)
- last_submit_time: last submission time (unix timestamp, seconds)
- end_time: task end time (unix timestamp, seconds)

Optional API:
- GET $GRADING_SERVER_URL/time
- Return fields: `start_time / timeout_s / last_submit_time / end_time / now / remaining`

## Workspace and Directory
**Your current directory is your workspace. All required files are in the current directory. **
- **Disable `cd` outside the current directory** (do not access parent directories or other paths)
- **Only use relative paths** (such as `./code/train.py`, not absolute paths)
- If you see a symlink pointing to an external path, ignore it - just use a relative path to access it.

Directory structure:
```
./
├── code/ # Your code area (all self-written code is placed here)
├── data/ # Training data (read-only)
├── models/ # Basic model (read-only)
├── output/ # Model output (the trained model is saved here)
├── description.md # Task description (required reading)
├── instructions.md # This document
├── run_meta.json # Time and budget signals (single source of truth)
└── ... # benchmark-specific files (use ls to view the complete list)
```

**First use `ls` to view all available files in the current directory. ** Different types of benchmarks will provide different additional files:
- **Interactive environment class** (such as ALFWorld): will provide `eval.py` (environment interaction + evaluation logic), prompt template, configuration file, etc. - these are key references for writing training code
- **Static data set class** (such as GSM8K): mainly provides training samples through data files under `data/`

**illustrate**:
- `code/`: write code here (filenames and structures freely organized)
- `output/`: The model storage location for training output. Can store multiple versions (such as `output/v1/`, `output/v2/`), specify the specific path when submitting

## Task process (earn points within a fixed time)
1. Explore the workspace, read `description.md`, `instructions.md` and related files (be sure to read `eval.py` carefully if there is one) to understand the task objectives and available resources
2. Write code under `code/` and train the model (SFT, GRPO, PPO, etc. are all acceptable)
3. Save the model to $OUTPUT_DIR (such as `output/v1`)
4. Submit a review: POST $GRADING_SERVER_URL/submit
5. Adjust the strategy based on the returned score, **Continue to iterate and submit a higher-scoring model in the remaining time**

## API
```bash
# Submit evaluation (specify model path, return score, improvement, best)
curl -X POST "$GRADING_SERVER_URL/submit" \
    -H "Content-Type: application/json" \
    -d '{"model_path": "'$OUTPUT_DIR'/v1"}'

# Specify GPU evaluation (optional, uses GPU 0 by default)
curl -X POST "$GRADING_SERVER_URL/submit" \
    -H "Content-Type: application/json" \
    -d '{"model_path": "'$OUTPUT_DIR'/v1", "gpu": "0"}'

# Multi-GPU review
curl -X POST "$GRADING_SERVER_URL/submit" \
    -H "Content-Type: application/json" \
    -d '{"model_path": "'$OUTPUT_DIR'/v1", "gpu": "2,3"}'

# Query time and budget (use run_meta.json first; this API is only a supplement)
curl "$GRADING_SERVER_URL/time"

# Health check (return available GPU list and other information)
curl "$GRADING_SERVER_URL/health"
```

### /submit parameters
| Parameters | Type | Required | Description |
|------|------|------|------|
| model_path | string | yes | model path |
| gpu | string | No | Specify GPU(s) (e.g., "0", "1", "0,1"); must be from available GPUs. If not passed, the first available GPU is used. Check available list via /health |

### /submit response example
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

## Notice
- **Do not directly copy/symlink the base model for submission** - an untrained base model will only get a baseline score (improvement = 0), which is a waste of submission opportunities. Must be trained before submitting.
- Different versions of the model can be submitted multiple times, and the system automatically tracks the highest score
- Make reasonable use of time and iteratively optimize based on score feedback
- **Full model must be submitted**: The evaluation system does not support the LoRA adapter directory. If you use LoRA/PEFT to train, you must merge before saving: `model = model.merge_and_unload(); model.save_pretrained(output_path); tokenizer.save_pretrained(output_path)`
- trl After saving the model, `extra_special_tokens` in `tokenizer_config.json` will be saved in list format, but vLLM/transformers requires dict format when loading. This field needs to be deleted after saving the model, otherwise the evaluation will fail.
