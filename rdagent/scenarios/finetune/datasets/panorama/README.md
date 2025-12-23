---
language:
- en
license: cc-by-nc-4.0
tags:
- patent
- legal
- retrieval
- classification
size_categories:
- 100K<n<1M
task_categories:
- text-classification
- question-answering
---

# PANORAMA Dataset

Patent examination benchmark capturing decision trails and rationales from [PANORAMA](https://huggingface.co/datasets/LG-AI-Research/PANORAMA).

**Repository**: [LG-AI-Research/PANORAMA](https://huggingface.co/datasets/LG-AI-Research/PANORAMA)

## Tasks

### 1. PAR4PC: Prior-Art Retrieval for Patent Claims

**Task**: Multi-label classification - select relevant prior-art documents from 8 candidates.

**Train samples**: 54,028

**Metrics**: Exact Match Accuracy, Custom Score (partial credit)

### 2. PI4PC: Paragraph Identification for Patent Claims

**Task**: Single-choice - identify the most relevant paragraph in a prior-art document.

**Train samples**: 64,210

**Metrics**: Exact Match Accuracy

### 3. NOC4PC: Novelty and Non-Obviousness Classification

**Task**: Ternary classification - determine if a claim should be ALLOW, 102 rejection, or 103 rejection.

**Train samples**: 136,211

**Metrics**: Macro F1-score, Per-class Accuracy

## Legal Background

- **35 U.S.C. §102 (Novelty)**: Claim rejected if anticipated by a single prior art reference
- **35 U.S.C. §103 (Non-Obviousness)**: Claim rejected if obvious from combining prior art

## Data Fields

Common fields across all tasks:

- `application_number`: Patent application identifier
- `claim_number`: The specific claim number being evaluated
- `context`: Patent document context (title, abstract, claims)
- `gold_answers`: Correct answers
- `silver_answers`: Partially correct answers
- `negative_answers`: Incorrect options

## CoT Quality Assessment

**IMPORTANT**: This dataset does NOT contain CoT annotations.

| Dimension | Value |
|-----------|-------|
| baseline_quality | N/A (no CoT) |
| task_type | legal reasoning |
| polish_difficulty | high |

**Baseline**: Raw data contains rejection reasons but NO step-by-step reasoning chains. Paper explicitly states *"lacked ground-truth CoTs"*. **You MUST generate CoT** for all samples before training.

## Baseline Performance (CoT Prompting)

| Task | Best Model | Score |
|------|-----------|-------|
| PAR4PC | Gemma-3-12B | 77.30% |
| PI4PC | GPT-4o | 62.62% |
| NOC4PC | Claude-3.7-Sonnet | 45.40% |

## Citation

```bibtex
@article{panorama2024,
  title={PANORAMA: A Dataset and Benchmarks Capturing Decision Trails and Rationales in Patent Examination},
  author={LG AI Research and KAIST},
  year={2024},
  url={https://huggingface.co/datasets/LG-AI-Research/PANORAMA}
}
```

## License

CC-BY-NC-4.0 License
