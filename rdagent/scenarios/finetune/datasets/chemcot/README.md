---
language:
- en
license: mit
tags:
- chemistry
- chain-of-thought
size_categories:
- 10K<n<100K
task_categories:
- text-generation
- question-answering
---

# ChemCoTBench: Chemical Reasoning Benchmark with Modular Operations

Dataset for the paper [Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations](https://arxiv.org/abs/2505.21318).


## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 23,223 |
| Samples with `struct_cot` | 20,081 |
| Samples with `raw_cot` | 13,597 |

### token statistics
For column `struct_cot`, the token statistics is:
{
  "empty_value_count": 0,
  "min": 36,
  "max": 1573,
  "p50": 301,
  "p99": 799
}

For column `raw_cot`, the token statistics is:
{
  "empty_value_count": 0,
  "min": 176,
  "max": 12367,
  "p50": 2163,
  "p99": 6357
}



## Tasks

The dataset comprises **4 main task categories**:

| Task | Subtasks | Description |
|------|----------|-------------|
| `mol_und` | `fg_count`, `ring_count`, `Murcko_scaffold`, `ring_system_scaffold` | Molecular understanding: functional group counting, ring counting, scaffold extraction |
| `mol_edit` | `add`, `delete`, `sub` | Molecular editing: structural modifications (addition, deletion, substitution) |
| `mol_opt` | `logp`, `solubility`, `qed`, `drd`, `gsk`, `jnk` | Molecular optimization: physicochemical (LogP, solubility, QED) and target-level (DRD2, GSK3-β, JNK3) |
| `rxn` | `fs_major_product`, `fs_by_product` | Reaction prediction: major product and by-product prediction |

## Data Format

Each sample contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `query` | string | The chemical problem/question |
| `task` | string | Main task: `mol_und`, `mol_edit`, `mol_opt`, or `rxn` |
| `subtask` | string | Specific subtask name |
| `struct_cot` | string | Structured chain-of-thought reasoning (available for ~87% samples) |
| `raw_cot` | string | Raw chain-of-thought annotation (available for ~59% samples) |
| `meta` | object | Additional metadata |

## Example
```json
{
  "id": "mol_opt_qed_001",
  "query": "Optimize the following molecule to improve its QED score: CC(C)Cc1ccc(C(C)C(=O)O)cc1",
  "task": "mol_opt",
  "subtask": "qed",
  "struct_cot": "Step 1: Parse the input SMILES and identify the molecular structure.\nStep 2: Analyze current QED-relevant properties.\nStep 3: Apply substitution operation to improve drug-likeness.\nStep 4: Validate the optimized structure.",
  "raw_cot": "Let me analyze this molecule step by step...",
  "meta": {}
}
```

## License

This dataset is released under the MIT License.

