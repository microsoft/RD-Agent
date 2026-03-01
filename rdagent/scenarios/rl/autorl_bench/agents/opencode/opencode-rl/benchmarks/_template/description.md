# GSM8K Benchmark

Train a language model to solve grade school math problems using GRPO reinforcement learning.

## Task
Given math word problems, the model should generate step-by-step solutions with correct final numerical answers.

## Data Format
Each sample has:
- `question`: A math word problem in natural language
- `answer`: Step-by-step solution with calculations marked by `<<...>>` and final answer after `####`

## Evaluation
Score is based on the percentage of correct final numerical answers on the test set.
