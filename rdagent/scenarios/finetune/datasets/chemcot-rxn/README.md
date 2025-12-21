---
language:
- en
license: mit
tags:
- chemistry
- chain-of-thought
- reaction-prediction
size_categories:
- 1K<n<10K
task_categories:
- text-generation
- question-answering
---

# ChemCoT Reaction Prediction (rxn)

Reaction prediction task from [ChemCoTBench](https://arxiv.org/abs/2505.21318).

## Subtasks

| Subtask | Description | Samples |
|---------|-------------|---------|
| `fs` | Forward synthesis (major product + by-product prediction) | 3,678 |
| `rcr` | Reaction Condition Recommendation (catalyst prediction) | 3,142 |

**Total: 6,820 samples**

## Evaluation Metrics

- Top-1 accuracy
- Fingerprint similarity

## Data Format

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `query` | string | The chemical problem/question |
| `task` | string | Always "rxn" |
| `subtask` | string | Specific subtask name (`fs` or `rcr`) |
| `struct_cot` | string | Structured chain-of-thought reasoning |
| `raw_cot` | string | Raw chain-of-thought annotation |
| `meta` | object | Additional metadata |
| `gt` | string | Ground truth catalyst (rcr subtask only) |

## Schema Unification

The original `rcr.json` uses `cot_result` instead of `struct_cot`. The `prepare.py` script automatically renames this column for consistency:

```python
# prepare.py
def prepare(ds: Dataset) -> Dataset:
    if "cot_result" in ds.column_names:
        ds = ds.rename_column("cot_result", "struct_cot")
    return ds
```

## License

MIT License
