#HumanEval Task

## Objective
The trained model achieved a higher pass@1 on HumanEval’s Python function completion task.

## Data format
```json
{
"question": "Function signature and docstring (prompt)",
"answer": "Reference implementation (canonical_solution)",
  "task_id": "HumanEval/0",
"entry_point": "Target function name",
"test": "Test code used to verify the correctness of the implementation"
}
```

## Evaluation indicators
- pass@1 (executed by OpenCompass HumanEval configuration)

## Data partition
- HumanEval original `test` has 164 entries in total.
- The visible data for training is fixed to the first 82 items (`[:82]`).
- The automatic evaluation is fixed to the last 82 items (`[82:]`) and does not overlap with the training set.

## hint
- Generate executable Python function implementations, giving priority to correctness.
- Note that the function name must be consistent with `entry_point`.
