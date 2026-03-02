---
language:
- en
size_categories:
- 10K<n<100K
license: mit
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  splits:
  - name: train
    num_examples: 40315
---

# DeepScaleR Mathematical Reasoning Dataset

Dataset for DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL.

> DeepScaleR-1.5B-Preview achieves **43.1% Pass@1 accuracy on AIME 2024**, representing a **15% improvement** over the base model (28.8%) and **surpassing OpenAI's O1-Preview performance** with just 1.5B parameters through distributed reinforcement learning.

## Overview

The **DeepScaleR dataset** is a carefully curated collection of approximately **40,000 unique mathematics problem-answer pairs** designed for training mathematical reasoning models through reinforcement learning. This dataset covers a wide range of competition-level mathematics problems from high school to olympiad level, providing a robust foundation for scaling RL algorithms on reasoning tasks.

DeepScaleR demonstrates that sophisticated mathematical reasoning can be achieved through strategic data curation combined with iterative context length scaling (8K→16K→24K) using Group Relative Policy Optimization (GRPO).


### Data Sources

Our training dataset consists of problems compiled from prestigious mathematics competitions and curated datasets:

- **AIME** (American Invitational Mathematics Examination) problems (1984-2023)
- **AMC** (American Mathematics Competition) problems (prior to 2023)
- **Omni-MATH** dataset
- **Still** dataset

### Data Fields

The dataset contains three key fields:

- `problem`: The mathematical problem statement, formatted with LaTeX notation
- `solution`: Official solution to the problem, including LaTeX formatting and boxed final answers. If there is no solution, the `solution` field is an empty string
- `answer`: The final mathematical result/answer, usually extracted from the solution

## CoT Quality Assessment

**IMPORTANT**: Raw data must be polished before training.

| Dimension | Value |
|-----------|-------|
| baseline_quality | low |
| task_type | math |
| polish_difficulty | high |

**Baseline**: 82% empty `solution`, 18% too short (p50=373 tokens, summary-style). Need to generate exploratory CoT (For your reference, the length of a well-structured CoT is usually longer than 1/4 * the model max_position_embeddings tokens) for all samples.

## License

This dataset is released under the MIT License.
