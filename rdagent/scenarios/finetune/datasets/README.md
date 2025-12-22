# Dataset Management Module

This module manages datasets for LLM Finetune scenarios, downloading entire HuggingFace repositories via `snapshot_download`.

## Design Goals

1. **Simplicity**: Download entire HF repo, preserve original file structure
2. **Extensibility**: Optional `post_download_fn` for custom processing (e.g., removing test splits)

## Usage

```python
from rdagent.scenarios.finetune.datasets import prepare, prepare_all, DATASETS

# 1. View registered datasets
print(DATASETS.keys())
# ['chemcot', 'panorama', 'deepscaler']

# 2. Prepare a dataset (download to local)
path = prepare("chemcot")
# Downloads to: datasets/chemcot/

# 3. Prepare all datasets
prepare_all()
```

## Dataset Configuration

Each dataset is configured via `DatasetConfig`:

```python
@dataclass
class DatasetConfig:
    repo_id: str                                          # HuggingFace repo ID
    post_download_fn: Optional[Callable[[str], None]]     # Post-download processing
```

## Registered Datasets

| Name | Repository | Description |
|------|------------|-------------|
| `chemcot` | OpenMol/ChemCoTDataset | Chemical reasoning with CoT |
| `panorama` | LG-AI-Research/PANORAMA | Patent examination benchmark |
| `deepscaler` | agentica-org/DeepScaleR-Preview-Dataset | Math reasoning |

## Adding New Datasets

Add configuration to `DATASETS` dict in `__init__.py`:

```python
DATASETS["my-dataset"] = DatasetConfig(
    repo_id="organization/dataset-name",
    post_download_fn=my_cleanup_function,  # Optional
)
```

Optionally add `datasets/my-dataset/README.md` for documentation.

## Directory Structure

```
datasets/
├── __init__.py          # Main module: prepare(), prepare_all(), DATASETS
├── README.md            # This document
├── chemcot/
│   └── README.md        # ChemCoT dataset documentation
├── panorama/
│   └── README.md        # PANORAMA dataset documentation
└── deepscaler/
    └── README.md        # DeepScaleR dataset documentation
```

## Notes

1. **Token**: Private datasets require `HF_TOKEN` environment variable
2. **Caching**: HuggingFace hub caches downloads automatically
3. **Force refresh**: Use `prepare(name, force=True)` to re-download
