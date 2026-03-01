# GSM8K - Grade School Math

Train a language model to solve grade school math word problems using GRPO (Group Relative Policy Optimization) reinforcement learning.

## Task
- Dataset: GSM8K (train split, JSONL format)
- Each sample has a `question` and `answer` field
- The answer ends with `#### <number>` which is the final numeric answer
- Train the model to generate step-by-step solutions that end with the correct `#### <number>`

## Requirements
- Use TRL's `GRPOTrainer` with LoRA (PEFT) for parameter-efficient training
- Implement a reward function that checks if the model's output contains the correct final answer
- The data path may be a directory containing JSONL files; handle both file and directory paths
- After training, **merge** the LoRA adapter into the base model (`model.merge_and_unload()`) and save the **full merged model** (not just the adapter) to the output directory. The output must contain `config.json` and `model.safetensors` so it can be loaded standalone by vLLM.

## Environment Variables
- `MODEL_PATH`: Path to the base model
- `DATA_PATH`: Path to training data (file or directory)
- `OUTPUT_DIR`: Where to save the trained model
- `MAX_STEPS`: Maximum training steps (default: 80)
- `MAX_TRAIN_SAMPLES`: Maximum training samples to use
