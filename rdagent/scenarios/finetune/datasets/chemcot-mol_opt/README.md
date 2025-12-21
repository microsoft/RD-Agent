---
language:
- en
license: mit
tags:
- chemistry
- chain-of-thought
- molecular-optimization
size_categories:
- 1K<n<10K
task_categories:
- text-generation
- question-answering
---

# ChemCoT Molecular Optimization (mol_opt)

Molecular optimization task from [ChemCoTBench](https://arxiv.org/abs/2505.21318).

## Subtasks

| Subtask | Description |
|---------|-------------|
| `logp` | LogP (lipophilicity) optimization |
| `solubility` | Aqueous solubility optimization |
| `qed` | QED (drug-likeness) optimization |
| `drd` | DRD2 binding affinity optimization |
| `gsk` | GSK3-beta binding affinity optimization |
| `jnk` | JNK3 binding affinity optimization |

## Evaluation Metrics

- Mean improvement rate
- Success rate (percentage of valid improvements)

## Data Format

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `query` | string | The chemical problem/question |
| `task` | string | Always "mol_opt" |
| `subtask` | string | Specific subtask name |
| `struct_cot` | string | Structured chain-of-thought reasoning |
| `raw_cot` | string | Raw chain-of-thought annotation |
| `meta` | object | Additional metadata |

## License

MIT License
