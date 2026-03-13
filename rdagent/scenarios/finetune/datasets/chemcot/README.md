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

## Overview

The **ChemCoTDataset** provides ~23K high-quality chain-of-thought samples for training chemical reasoning models. CoT annotations were distilled from state-of-the-art reasoning models (Gemini-2.5-pro, DeepSeek-R1, Claude-3.7-sonnet-thinking) and validated by 13 chemistry PhD candidates with >90% accuracy.

### Dataset Scale

| Category | Subtasks | Samples |
|----------|----------|---------|
| mol_und | fg_count, ring_count, ring_system_scaffold, Murcko_scaffold | 6,319 |
| mol_edit | add, delete, sub | 4,497 |
| mol_opt | drd, gsk, jnk, qed, solubility, logp | 5,587 |
| rxn | fs_by_product, fs_major_product, rcr | 6,820 |
| **Total** | **16 subtasks** | **23,223** |

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

## CoT Quality Assessment

**IMPORTANT**: Distilled CoT may require domain refinement.

| Dimension | Value |
|-----------|-------|
| baseline_quality | medium-high |
| task_type | chemistry |
| polish_difficulty | medium |

**Baseline**: CoT distilled from Gemini-2.5-pro/DeepSeek-R1/Claude, validated by 13 chemistry PhD candidates (>90% accuracy). Paper notes: *"distillation strategy falters in chemistry"* - consider expert refinement for optimal results.

## License

MIT License
