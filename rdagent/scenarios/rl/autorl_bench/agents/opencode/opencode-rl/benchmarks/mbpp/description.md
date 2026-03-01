# MBPP Benchmark

Train a language model to solve basic Python programming problems using GRPO reinforcement learning.

## Task
Given a natural language description of a programming task, the model should generate correct Python code.

## Data Format
Each sample in `train.jsonl` has:
- `question`: Natural language description of the programming task
- `answer`: Reference implementation (correct Python code)
- `task_id`: Unique numeric identifier
- `test_list`: A list of assertion strings (e.g., `["assert func(1)==2", "assert func(3)==4"]`)

## Evaluation
- For each sample, the model generates a completion (Python code)
- The completion is tested by executing each assertion in `test_list` against the generated code
- If all assertions pass, reward=1.0; otherwise reward=0.0
- Score = pass@1 = (number of passed samples) / (total samples)

## Important Notes
- The `test_list` field is a list of strings, each containing an `assert` statement
- To verify correctness: exec() the model's generated code, then exec() each assert string
- If all asserts pass without exception, the code is correct
- Each test execution should have a timeout (5 seconds) to prevent infinite loops
