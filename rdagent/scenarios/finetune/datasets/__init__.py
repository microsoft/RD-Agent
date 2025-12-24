"""Dataset preparation module for finetune scenarios.

Usage:
    from rdagent.scenarios.finetune.datasets import prepare, prepare_all

    prepare("chemcot")     # Download ChemCoT dataset
    prepare("panorama")    # Download PANORAMA dataset
    prepare_all()          # Prepare all registered datasets
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.scenarios.finetune.datasets.chemcot import normalize_rcr
from rdagent.scenarios.finetune.datasets.financeiq.split import split_financeiq_dataset
from rdagent.scenarios.finetune.download.hf import download_dataset


@dataclass
class DatasetConfig:
    """Configuration for a registered dataset.

    Attributes:
        repo_id: HuggingFace dataset repository ID
        post_download_fn: Optional function to run after download (e.g., remove test split)
    """

    repo_id: str
    post_download_fn: Optional[Callable[[str], None]] = field(default=None)


def _remove_eval_splits(out_dir: str) -> None:
    """Remove validation and test split files to prevent data leakage."""
    for pattern in ["*validation*", "*test*"]:
        for f in Path(out_dir).rglob(pattern):
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)


# Dataset registry: name -> DatasetConfig
DATASETS: dict[str, DatasetConfig] = {
    "chemcot": DatasetConfig(
        repo_id="OpenMol/ChemCoTDataset",
        post_download_fn=normalize_rcr,
    ),
    "panorama": DatasetConfig(
        repo_id="LG-AI-Research/PANORAMA",
        post_download_fn=_remove_eval_splits,
    ),
    "deepscaler": DatasetConfig(
        repo_id="agentica-org/DeepScaleR-Preview-Dataset",
    ),
    "financeiq": DatasetConfig(
        repo_id="Duxiaoman-DI/FinanceIQ",
        post_download_fn=lambda out_dir: split_financeiq_dataset(out_dir, split="train"),
    ),
    "tableinstruct": DatasetConfig(
        repo_id="Multilingual-Multimodal-NLP/TableInstruct",
    )
}


def prepare(name: str, force: bool = False) -> str:
    """Download dataset to local directory using snapshot_download.

    Downloads the entire HuggingFace dataset repository, preserving the original
    file structure.

    Args:
        name: Dataset name (must be registered in DATASETS)
        force: If True, re-download even if exists

    Returns:
        Path to the dataset directory
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    config = DATASETS[name]
    out_dir = Path(FT_RD_SETTING.file_path) / "datasets" / name

    # Skip if already exists and not forcing
    if not force and out_dir.exists():
        return str(out_dir)

    # Download using snapshot_download
    download_dataset(
        repo_id=config.repo_id,
        out_dir=str(out_dir),
        force=force,
    )

    # Run post-download processing if defined
    if config.post_download_fn:
        config.post_download_fn(str(out_dir))

    # Copy custom README if exists in source code
    custom_readme = Path(__file__).parent / name / "README.md"
    if custom_readme.exists():
        shutil.copy(custom_readme, out_dir / "README.md")

    return str(out_dir)


def prepare_all(force: bool = False) -> dict[str, str]:
    """Prepare all registered datasets.

    Args:
        force: If True, re-download even if exists

    Returns:
        Dict mapping dataset name to download path
    """
    return {name: prepare(name, force=force) for name in DATASETS}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        path = prepare(dataset_name)
        print(f"Dataset prepared at: {path}")
    else:
        print(f"Available datasets: {list(DATASETS.keys())}")
