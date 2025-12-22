---
language:
- en
license: mit
tags:
- chemistry
- chain-of-thought
- molecular-reasoning
size_categories:
- 10K<n<100K
task_categories:
- text-generation
- question-answering
---

# ChemCoT Dataset

Chemical reasoning dataset with Chain-of-Thought annotations from [ChemCoTBench](https://arxiv.org/abs/2505.21318).

**Repository**: [OpenMol/ChemCoTDataset](https://huggingface.co/datasets/OpenMol/ChemCoTDataset)

## Tasks

### 1. Molecular Understanding (mol_und)

| Subtask | Description |
|---------|-------------|
| `fg_count` | Functional group counting |
| `ring_count` | Ring counting |
| `Murcko_scaffold` | Murcko scaffold extraction |
| `ring_system_scaffold` | Ring system scaffold extraction |

**Metrics**: MAE for counting, Tanimoto similarity for scaffold extraction

### 2. Molecular Editing (mol_edit)

| Subtask | Description |
|---------|-------------|
| `add` | Add functional groups to molecules |
| `delete` | Delete functional groups from molecules |
| `sub` | Substitute functional groups in molecules |

**Metrics**: Pass@1 (validity and instruction matching)

### 3. Molecular Optimization (mol_opt)

| Subtask | Description |
|---------|-------------|
| `logp` | LogP (lipophilicity) optimization |
| `solubility` | Aqueous solubility optimization |
| `qed` | QED (drug-likeness) optimization |
| `drd` | DRD2 binding affinity optimization |
| `gsk` | GSK3-beta binding affinity optimization |
| `jnk` | JNK3 binding affinity optimization |

**Metrics**: Mean improvement rate, Success rate

### 4. Reaction Prediction (rxn)

| Subtask | Description |
|---------|-------------|
| `fs` | Forward synthesis (major product + by-product prediction) |
| `rcr` | Reaction Condition Recommendation (catalyst prediction) |

**Metrics**: Top-1 accuracy, Fingerprint similarity

## Data Format

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `query` | string | The chemical problem/question |
| `task` | string | Task category (mol_und, mol_edit, mol_opt, rxn) |
| `subtask` | string | Specific subtask name |
| `struct_cot` | string | Structured chain-of-thought reasoning |
| `raw_cot` | string | Raw chain-of-thought annotation |
| `meta` | object | Additional metadata |

## License

MIT License
