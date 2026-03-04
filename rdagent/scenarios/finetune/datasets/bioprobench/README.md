---
license: cc-by-4.0
configs:
- config_name: PQA
	data_files:
		- split: train
			path: PQA.json
		- split: test
			path: PQA_test.json
- config_name: ERR
	data_files:
		- split: train
			path: ERR.json
		- split: test
			path: ERR_test.json
- config_name: ORD
	data_files:
		- split: train
			path: ORD.json
		- split: test
			path: ORD_test.json
- config_name: GEN
	data_files:
		- split: train
			path: GEN.json
		- split: test
			path: GEN_test.json
---

# BioProBench Dataset for LLM Fine-Tuning

BioProBench is a large-scale, multi-task benchmark focused on biological protocol understanding and reasoning for large language models (LLMs). It spans four fine-tuning tasks provided here: Protocol Question Answering (PQA), Step Ordering (ORD), Error Correction (ERR), and Protocol Generation (GEN).

This dataset is built on a raw corpus of ~27K biological protocols and provides over 550K structured instances across tasks, with a held-out test set of 1,000 examples per task. See the original benchmark for full details:
- Code: https://github.com/YuyangSunshine/bioprotocolbench/
- Dataset hub: https://huggingface.co/BioProBench
- License: CC BY 4.0

## Data Files

The JSON files for each task (train/test) are organized per task. If your fine-tuning pipeline expects local files, place them alongside this README or update paths accordingly.

- PQA: [bioprobench/PQA.json](bioprobench/PQA.json), [bioprobench/PQA_test.json](bioprobench/PQA_test.json)
- ERR: [bioprobench/ERR.json](bioprobench/ERR.json), [bioprobench/ERR_test.json](bioprobench/ERR_test.json)
- ORD: [bioprobench/ORD.json](bioprobench/ORD.json), [bioprobench/ORD_test.json](bioprobench/ORD_test.json)
- GEN: [bioprobench/GEN.json](bioprobench/GEN.json), [bioprobench/GEN_test.json](bioprobench/GEN_test.json)

## Task Definitions and Fields

### PQA — Protocol Question Answering
Multiple-choice QA over protocol content.
- Fields:
	- `question`: the question string
	- `choices`: list of candidate answers
	- `answer`: the correct answer
	- `type`: category of the question (e.g., parameter, reagent, operation)
	- `id`: unique identifier

### ORD — Step Ordering
Order protocol steps correctly (top-level or sub-step sequences).
- Fields:
	- `question`: prompt describing the step list and context/title
	- `wrong_steps`: list of steps in a shuffled or incorrect order
	- `correct_steps`: steps in the correct chronological order
	- `type`: sequence granularity (e.g., `top`, `child`)
	- `id`: unique identifier

### ERR — Error Correction
Detect and correct errors in protocol text with local context.
- Fields:
	- `context`: object with `purpose`, `prior_step`, `next_step`
	- `corrupted_text`: the erroneous text (may be `null` for correct cases)
	- `corrected_text`: corrected version of the text
	- `is_correct`: boolean indicating whether the provided text was already correct
	- `type`: category (e.g., parameter, reagent, operation, or `correct`)
	- `error_description`: brief rationale for the correction
	- `id`: unique identifier

### GEN — Protocol Generation
Generate concise, single-level, numbered protocol steps from prompts.
- Fields:
	- `system_prompt`: role/system instruction
	- `instruction`: formatting and style constraints
	- `input`: task description or query
	- `output`: list of numbered steps (flat 1., 2., 3. ...)
	- `id`: unique identifier
	- `type`: difficulty tag (e.g., `easy`)

## Splits
- Train: use the non-`_test.json` files per task.
- Test: each task provides a held-out set of 1,000 examples.

## Training Data Guidelines (CRITICAL for Fine-tuning)

### ERR — Error Correction

**CRITICAL: Answer Semantics**

The benchmark prompt says: "If you find anything wrong, answer False."

| Condition | `is_correct` field | Correct training output |
|-----------|-------------------|------------------------|
| Protocol step is CORRECT | `True` | `[ANSWER_START]True[ANSWER_END]` |
| Protocol step HAS ERRORS | `False` | `[ANSWER_START]False[ANSWER_END]` |

**Important**: The training data generation script MUST use this logic:
```python
def gold_answer_from_is_correct(is_correct: bool) -> str:
    # True = step is correct, False = step has errors
    return ANSWER_TRUE if is_correct else ANSWER_FALSE
```

Do NOT invert this logic - the benchmark evaluator compares model output directly with `is_correct` field.

### ORD — Step Ordering

**Output Format (CRITICAL)**
- Answer MUST be a valid Python list: `[0, 2, 1, 3]`
- NOT space-separated: `0 2 1 3` (WRONG)
- NOT comma-separated without brackets: `0, 2, 1, 3` (WRONG)

**Training Data Format**
- Can include brief reasoning (1-2 sentences)
- Final answer MUST be in format: `[ANSWER_START][list][ANSWER_END]`
- Example: `[ANSWER_START][2, 0, 1, 3][ANSWER_END]`

### GEN — Protocol Generation

**Output Format**
- Step-by-step protocol wrapped in `[ANSWER_START]...[ANSWER_END]`
- CoT (Chain-of-Thought) format is acceptable for this task

### Common Notes
- All tasks support `<think>...</think>` tags for CoT reasoning (evaluator will strip them)
- Answer MUST be wrapped in `[ANSWER_START]` and `[ANSWER_END]` tags

## License
- CC BY 4.0 — see https://creativecommons.org/licenses/by/4.0/

## Notes
- Tasks cover protocol QA, ordering, correction, and generation (REA is part of the broader benchmark but not included in the files above).
- Data spans diverse biological domains and repositories; see the original benchmark for details.
