---
language:
- en
size_categories:
- 100K<n<1M
license: cc-by-nc-4.0
configs:
- config_name: NOC4PC
  data_files:
  - split: train
    path: NOC4PC/train.parquet
  - split: validation
    path: NOC4PC/validation.parquet
  splits:
  - name: train
    num_examples: 136211
  - name: validation
    num_examples: 2884
---

# PANORAMA - NOC4PC: Novelty and Non-Obviousness Classification for Patent Claims

Dataset for patent novelty and non-obviousness classification from the PANORAMA benchmark.

> NOC4PC is a **ternary classification task** where the goal is to determine whether a patent claim should be rejected under §102 (lack of novelty), §103 (obviousness), or be allowed.

## Overview

The **NOC4PC (Novelty and Non-Obviousness Classification for Patent Claims)** task evaluates a model's ability to make patentability determinations. Given a patent claim and relevant prior-art documents with identified paragraphs, the model must classify whether the claim should be:

1. **ALLOW**: The claim is novel and non-obvious
2. **102 Rejection**: The claim lacks novelty (anticipated by prior art)
3. **103 Rejection**: The claim is obvious in view of prior art

This task is the final step in patent examination, requiring deep understanding of both the claim and the prior art.

## Task Definition

- **Input**: A patent claim and cited prior-art documents with relevant paragraphs
- **Output**: Classification decision (ALLOW, 102, or 103)
- **Format**: Ternary classification

## Data Fields

The dataset contains the following key fields:

- `application_number`: Patent application identifier
- `claim_number`: The specific claim number being evaluated
- `context`: Patent document context including:
  - `title`: Patent title
  - `abstract`: Patent abstract
  - `claims`: List of patent claims
- `prior_art_specifications`: Relevant prior art text with identified paragraphs
- `answer`: Classification label (ALLOW, 102, or 103)

## Example

```json
{
  "application_number": "US12345678",
  "claim_number": 1,
  "context": {
    "title": "Improved battery charging system",
    "abstract": "A system for efficiently charging batteries...",
    "claims": ["A charging system comprising...", "The system of claim 1..."]
  },
  "prior_art_specifications": "[0045] A charging circuit that monitors voltage levels... [0078] The controller adjusts current based on...",
  "answer": "103"
}
```

## Evaluation Metrics

- **Macro F1-score**: Primary metric for handling class imbalance
- **Per-class Accuracy**: Accuracy for each classification category

## Legal Background

- **35 U.S.C. §102 (Novelty)**: A claim is rejected if it is anticipated by a single prior art reference that discloses every element of the claim
- **35 U.S.C. §103 (Non-Obviousness)**: A claim is rejected if it would have been obvious to combine multiple prior art references

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

This dataset is released under the CC-BY-NC-4.0 License.
