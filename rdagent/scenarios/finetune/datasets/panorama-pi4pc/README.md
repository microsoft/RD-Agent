---
language:
- en
size_categories:
- 10K<n<100K
license: cc-by-nc-4.0
configs:
- config_name: PI4PC
  data_files:
  - split: train
    path: PI4PC/train.parquet
  - split: validation
    path: PI4PC/validation.parquet
  splits:
  - name: train
    num_examples: 64210
  - name: validation
    num_examples: 3402
---

# PANORAMA - PI4PC: Paragraph Identification for Patent Claims

Dataset for identifying relevant paragraphs in prior-art documents from the PANORAMA benchmark.

> PI4PC is a **single-choice task** where the goal is to identify the most relevant paragraph in a prior-art document that should be compared with a patent claim when assessing patentability.

## Overview

The **PI4PC (Paragraph Identification for Patent Claims)** task evaluates a model's ability to pinpoint the specific portion of a prior-art document that is most relevant to a given patent claim. This is a crucial step in patent examination, as examiners must cite specific paragraphs when rejecting claims.

Given a patent claim and a prior-art document, the model must select the paragraph that provides the strongest basis for comparing with the claim.

## Task Definition

- **Input**: A patent claim and a prior-art document with 5 candidate paragraphs
- **Output**: Selection of the most relevant paragraph (single choice)
- **Format**: Single-choice question with gold answer being the examiner-cited paragraph

## Data Fields

The dataset contains the following key fields:

- `application_number`: Patent application identifier
- `claim_number`: The specific claim number being evaluated
- `context`: Patent document context including:
  - `title`: Patent title
  - `abstract`: Patent abstract
  - `claims`: List of patent claims
- `prior_art_specification`: The prior-art document content
- `options`: 5 candidate paragraphs (A-E) from the prior-art document
  - Each option contains paragraph text from different sections
- `gold_answers`: The correct paragraph(s) as cited by the examiner
- `silver_answers`: Alternative acceptable paragraphs
- `negative_answers`: Incorrect options

## Example

```json
{
  "application_number": "US12345678",
  "claim_number": 1,
  "context": {
    "title": "Wireless communication protocol",
    "abstract": "A method for wireless data transmission...",
    "claims": ["A method comprising transmitting data...", "The method of claim 1..."]
  },
  "options": {
    "A": "[0023] The transmitter module includes...",
    "B": "[0045] Data packets are formatted according to...",
    "C": "[0067] The receiver decodes signals by...",
    "D": "[0089] Power management controls include...",
    "E": "[0112] Error correction is performed using..."
  },
  "gold_answers": ["B"],
  "silver_answers": [],
  "negative_answers": ["A", "C", "D", "E"]
}
```

## Evaluation Metrics

- **Exact Match Accuracy**: Percentage of correct paragraph selections
- **Custom Score**: Considers partial credit for silver answers

## Importance

Accurate paragraph identification is essential for:
1. **Transparency**: Examiners must cite specific evidence for rejections
2. **Efficiency**: Reduces time spent reviewing irrelevant sections
3. **Appeals**: Provides clear basis for applicant responses

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
