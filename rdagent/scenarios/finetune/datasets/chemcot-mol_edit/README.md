---
language:
- en
license: mit
tags:
- chemistry
- chain-of-thought
- molecular-editing
size_categories:
- 1K<n<10K
task_categories:
- text-generation
- question-answering
---

# ChemCoT Molecular Editing (mol_edit)

Molecular editing task from [ChemCoTBench](https://arxiv.org/abs/2505.21318).

## Subtasks

| Subtask | Description |
|---------|-------------|
| `add` | Add functional groups to molecules |
| `delete` | Delete functional groups from molecules |
| `sub` | Substitute functional groups in molecules |

## Evaluation Metrics

- Pass@1: Whether the generated molecule is chemically valid and matches the instruction

## Data Format

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `query` | string | The chemical problem/question |
| `task` | string | Always "mol_edit" |
| `subtask` | string | Specific subtask name |
| `struct_cot` | string | Structured chain-of-thought reasoning |
| `raw_cot` | string | Raw chain-of-thought annotation |
| `meta` | object | Additional metadata |

## License

MIT License
