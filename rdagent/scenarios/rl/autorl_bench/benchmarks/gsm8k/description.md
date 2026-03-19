#GSM8K task

## Target
Train the model to achieve higher accuracy on GSM8K math problems.

## Data format
```json
{"question": "...", "answer": "... #### 42"}
```

## Evaluation indicators
- Answer accuracy (exact match)

## hint
- Answer format: `#### number`
- Training using RL methods such as GRPO/PPO
