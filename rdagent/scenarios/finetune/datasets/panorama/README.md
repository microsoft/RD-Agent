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

## Data Format (Parquet Fields)

### PAR4PC / PI4PC Format

| Field | Type | Description |
|-------|------|-------------|
| `application_number` | str | Patent application identifier |
| `claim_number` | int64 | Specific claim number being evaluated |
| `context` | dict | Patent context: `{abstract: str, claims: list[str], title: str}` |
| `options` | dict | 8 candidate documents: `{A: {abstract, claims, patent_id, title}, B: {...}, ...}` |
| `gold_answers` | ndarray | Correct answer labels, e.g. `array(['G'])` or `array(['A', 'C'])` |
| `silver_answers` | ndarray | Partially correct answers |
| `negative_answers` | ndarray | Incorrect options |

**Note**: PI4PC has an additional `prior_art_specification` field containing the relevant prior-art document text.

### NOC4PC Format

| Field | Type | Description |
|-------|------|-------------|
| `application_number` | str | Patent application identifier |
| `claim_number` | int64 | Specific claim number being evaluated |
| `context` | dict | Patent context: `{abstract: str, claims: list[str], title: str}` |
| `prior_art_specifications` | list | Prior art document specifications |
| `answer` | str | Classification label: `ALLOW`, `102`, or `103` |

**Important**: Array fields (`gold_answers`, `silver_answers`, `negative_answers`) are `numpy.ndarray` type.
Use `.tolist()` to convert to Python list before processing.

### Example Data

```python
{
    "application_number": 14281639,
    "claim_number": 1,
    "context": {
        "abstract": "In an endodontic procedure...",
        "claims": ["claim 1 text", "claim 2 text", ...],
        "title": "Method for irrigating root canals"
    },
    "options": {
        "A": {"abstract": "...", "claims": [...], "patent_id": "US1234567", "title": "..."},
        "B": {"abstract": "...", "claims": [...], "patent_id": "US2345678", "title": "..."},
        # ... G, H
    },
    "gold_answers": array(['G'], dtype=object),  # numpy.ndarray, use .tolist() -> ['G']
    "negative_answers": array(['A', 'B', 'C', 'D', 'E', 'F', 'H'], dtype=object)
}
```

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
