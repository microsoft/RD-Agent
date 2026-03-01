# HumanEval Benchmark

Train a language model to solve Python programming problems using GRPO reinforcement learning.

## Task
Given a function signature and docstring, the model should generate a correct Python function implementation.

## Data Format
Each sample in `train.jsonl` has:
- `question`: Function signature with docstring describing the expected behavior
- `answer`: Reference implementation (correct Python code)
- `task_id`: Unique identifier like "HumanEval/0"
- `entry_point`: The function name to be implemented (e.g., "has_close_elements")
- `test`: A check function that tests the implementation with assertions

## Evaluation
- For each sample, the model generates a completion (function body)
- The completion is combined with the test cases from the `test` field
- Code is executed: if all assertions pass, reward=1.0; otherwise reward=0.0
- Score = pass@1 = (number of passed samples) / (total samples)

## Important Notes
- The `test` field contains a function like `check(candidate)` that runs assertions against the generated function
- To verify correctness: concatenate the model's generated code + the `test` code, then exec() it
- If exec() raises no exception, the code is correct
