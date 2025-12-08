"""Dataset preparation module for finetune scenarios.

Usage:
    from rdagent.scenarios.finetune.datasets import prepare, prepare_all

    prepare("deepscaler")   # Prepare single dataset
    prepare_all()           # Prepare all registered datasets
"""

import shutil
from pathlib import Path

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.scenarios.finetune.download.hf import download_dataset

# Dataset registry: name -> HuggingFace repo_id
# Empty for debug: use local limo dataset instead of downloading
DATASETS = {
    "deepscaler": "agentica-org/DeepScaleR-Preview-Dataset", # Removed for debug(Need to prepare data manually)
}


def prepare(name: str, force: bool = False) -> str:
    """Download dataset and apply custom README.

    Args:
        name: Dataset name (must be registered in DATASETS)
        force: If True, re-download even if exists

    Returns:
        Path to downloaded dataset directory
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    repo_id = DATASETS[name]
    out_dir_root = Path(FT_RD_SETTING.file_path) / "datasets"
    save_path = out_dir_root / repo_id

    if not force and save_path.exists():
        return str(save_path)

    download_dataset(repo_id, out_dir_root=str(out_dir_root), force=force)

    # Copy custom README (overwrite if exists)
    custom_readme = Path(__file__).parent / name / "README.md"
    if custom_readme.exists():
        shutil.copy(custom_readme, Path(save_path) / "README.md")

    return save_path


def prepare_all(force: bool = False) -> dict[str, str]:
    """Prepare all registered datasets.

    Args:
        force: If True, re-download even if exists

    Returns:
        Dict mapping dataset name to download path
    """
    return {name: prepare(name, force=force) for name in DATASETS}


if __name__ == "__main__":
    path = prepare("deepscaler")
    print(f"Dataset prepared at: {path}")
