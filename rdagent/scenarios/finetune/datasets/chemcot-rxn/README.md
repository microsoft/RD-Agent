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

| Subtask | Description |
|---------|-------------|
| `fs_major_product` | Forward synthesis major product prediction |
| `fs_by_product` | Forward synthesis by-product prediction |

## Evaluation Metrics

- Top-1 accuracy
- Fingerprint similarity

## Data Format

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `query` | string | The chemical problem/question |
| `task` | string | Always "rxn" |
| `subtask` | string | Specific subtask name |
| `struct_cot` | string | Structured chain-of-thought reasoning |
| `raw_cot` | string | Raw chain-of-thought annotation |
| `meta` | object | Additional metadata |

## License

MIT License
