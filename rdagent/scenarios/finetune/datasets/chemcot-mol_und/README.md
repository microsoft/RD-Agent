---
language:
- en
license: mit
tags:
- chemistry
- chain-of-thought
- molecular-understanding
size_categories:
- 1K<n<10K
task_categories:
- text-generation
- question-answering
---

# ChemCoT Molecular Understanding (mol_und)

Molecular understanding task from [ChemCoTBench](https://arxiv.org/abs/2505.21318).

## Subtasks

| Subtask | Description |
|---------|-------------|
| `fg_count` | Functional group counting |
| `ring_count` | Ring counting |
| `Murcko_scaffold` | Murcko scaffold extraction |
| `ring_system_scaffold` | Ring system scaffold extraction |

## Evaluation Metrics

- MAE (Mean Absolute Error) for counting tasks
- Tanimoto similarity for scaffold extraction

## Data Format

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `query` | string | The chemical problem/question |
| `task` | string | Always "mol_und" |
| `subtask` | string | Specific subtask name |
| `struct_cot` | string | Structured chain-of-thought reasoning |
| `raw_cot` | string | Raw chain-of-thought annotation |
| `meta` | object | Additional metadata |

## License

MIT License
