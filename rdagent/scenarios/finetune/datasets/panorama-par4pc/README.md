---
language:
- en
size_categories:
- 10K<n<100K
license: cc-by-nc-4.0
configs:
- config_name: PAR4PC
  data_files:
  - split: train
    path: PAR4PC/train.parquet
  - split: validation
    path: PAR4PC/validation.parquet
  splits:
  - name: train
    num_examples: 54028
  - name: validation
    num_examples: 2896
---

# PANORAMA - PAR4PC: Prior-Art Retrieval for Patent Claims

Dataset for patent prior-art retrieval from the PANORAMA benchmark.

> PAR4PC is a multi-label classification task where the goal is to **select the relevant prior-art documents** from a pool of 8 candidates that should be consulted when determining whether a patent claim should be rejected.

## Overview

The **PAR4PC (Prior-Art Retrieval for Patent Claims)** task is designed to evaluate a model's ability to identify relevant prior art for patent examination. Given a patent claim and a set of candidate prior-art documents, the model must determine which documents are most relevant for assessing the patentability of the claim.

This task emulates the work of patent examiners who must search through existing patents to find relevant prior art that could affect the novelty or non-obviousness of a new patent application.

## Task Definition

- **Input**: A patent claim context (title, abstract, claims) and 8 candidate prior-art documents
- **Output**: Selection of relevant prior-art documents (multi-label)
- **Format**: Multiple choice with gold (correct) and silver (partially correct) answers

## Data Fields

The dataset contains the following key fields:

- `application_number`: Patent application identifier
- `claim_number`: The specific claim number being evaluated
- `context`: Patent document context including:
  - `title`: Patent title
  - `abstract`: Patent abstract
  - `claims`: List of patent claims
- `options`: 8 candidate prior-art documents (A-H), each containing:
  - `patent_id`: Prior art patent ID
  - `title`: Prior art title
  - `abstract`: Prior art abstract
  - `claims`: Prior art claims
- `gold_answers`: Primary correct answers
- `silver_answers`: Partially correct answers
- `negative_answers`: Incorrect options

## Example

```json
{
  "application_number": "US12345678",
  "claim_number": 1,
  "context": {
    "title": "Method for data compression",
    "abstract": "A novel method for compressing data...",
    "claims": ["A method comprising...", "The method of claim 1..."]
  },
  "options": {
    "A": {"patent_id": "US9876543", "title": "...", "abstract": "...", "claims": [...]},
    "B": {"patent_id": "US8765432", "title": "...", "abstract": "...", "claims": [...]},
    ...
  },
  "gold_answers": ["A", "C"],
  "silver_answers": ["E"],
  "negative_answers": ["B", "D", "F", "G", "H"]
}
```

## Evaluation Metrics

- **Exact Match Accuracy**: Percentage of instances where the model's selection exactly matches the gold answers
- **Custom Score**: Considers partial credit for silver answers

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
