---
language:
- zh
license: cc-by-nc-sa-4.0
tags:
- finance
- chinese
- multiple-choice
- professional-certification
size_categories:
- 1K<n<10K
task_categories:
- question-answering
- text-generation
---

# FinanceIQ Dataset

Chinese financial professional certification exam questions covering 10 major financial domains.

**Repository**: [LlamaFactory/FinanceIQ](https://huggingface.co/datasets/LlamaFactory/FinanceIQ)

## Overview

The **FinanceIQ dataset** is a comprehensive collection of approximately **6,179 multiple-choice questions** from Chinese financial professional certification exams. It covers 10 distinct financial domains, providing a robust benchmark for evaluating financial reasoning capabilities in Chinese language models.

### Dataset Scale

| Category | Chinese Name | Samples |
|----------|--------------|---------|
| Insurance (CICE) | 保险从业资格CICE | 596 |
| Fund Practitioner | 基金从业资格 | 772 |
| Futures Practitioner | 期货从业资格 | 333 |
| CPA | 注册会计师（CPA） | 1,211 |
| Financial Planner | 理财规划师 | 195 |
| Tax Advisor | 税务师 | 388 |
| Actuary (Financial Math) | 精算师-金融数学 | 44 |
| Economist | 经济师 | 420 |
| Securities Practitioner | 证券从业资格 | 1,076 |
| Banking Practitioner | 银行从业资格 | 1,144 |
| **Total** | **10 categories** | **6,179** |

## Tasks

### Single-Choice Question Answering

**Task**: Select the correct answer (A/B/C/D) from four options for each financial question.

**Evaluation**: LLM Judge (comparing model's answer selection with ground truth)

**Metrics**: Accuracy per category, Average accuracy

## Data Format (CSV Fields)

| Field | Type | Description |
|-------|------|-------------|
| `Question` | string | The question text in Chinese |
| `A` | string | Option A text |
| `B` | string | Option B text |
| `C` | string | Option C text |
| `D` | string | Option D text |
| `Answer` | string | Correct answer (A, B, C, or D) |

### Example Data

```csv
Question,A,B,C,D,Answer
关于生命价值理论的理解，以下哪一项表述是不正确的？,补偿生命经济价值可能受到的损失...,个人预期收入的货币价值...,任何触及个人收入能力的事件...,早逝、残疾、退休或失业可能导致...,B
```

## Data Split Strategy

The dataset uses an end-based split strategy:

- **Test set**: Takes from the END of each category (up to 50 samples per category, or 50% if fewer)
- **Train set**: Takes the remaining samples from the START

This ensures consistent train/test separation across all categories.

## Category Distribution Analysis

**Important**: Sample distribution is highly imbalanced:

| Category | Samples | % of Total | Note |
|----------|---------|------------|------|
| CPA | 1,211 | 19.6% | Largest |
| Banking | 1,144 | 18.5% | |
| Securities | 1,076 | 17.4% | |
| Fund | 772 | 12.5% | |
| Insurance | 596 | 9.6% | |
| Economist | 420 | 6.8% | |
| Tax | 388 | 6.3% | |
| Futures | 333 | 5.4% | |
| Financial Planner | 195 | 3.2% | Small |
| **Actuary** | **44** | **0.7%** | **Critically small** |

**Recommendation**: When generating training data, ensure balanced sampling across categories, especially for Actuary (精算师) which has only 44 samples.

## CoT Quality Assessment

**IMPORTANT**: Raw data contains only Q&A pairs, no reasoning chains.

| Dimension | Value |
|-----------|-------|
| baseline_quality | N/A (no CoT) |
| task_type | finance reasoning |
| polish_difficulty | medium |

**Baseline**: Questions are multiple-choice format without explanations. **You MUST generate CoT** (chain-of-thought reasoning) for training samples to achieve good results.

## Baseline Performance

| Model | Accuracy |
|-------|----------|
| Qwen2.5-7B-Instruct (zero-shot) | ~65% |

**Note**: The Actuary (精算师-金融数学) category is particularly challenging, with baseline accuracy around 27-36%.

## License

CC-BY-NC-SA-4.0 License
