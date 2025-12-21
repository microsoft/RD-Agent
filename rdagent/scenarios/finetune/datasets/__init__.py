"""Dataset preparation module for finetune scenarios.

Usage:
    from rdagent.scenarios.finetune.datasets import prepare, prepare_all, load_split

    prepare("panorama-par4pc")   # Download train split only (safe, no test leakage)
    prepare("chemcot-mol_opt")   # Prepare ChemCoT molecular optimization task
    prepare_all()                # Prepare all registered datasets
    load_split("panorama-par4pc")  # Load as Dataset object
"""

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from datasets import Dataset

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.scenarios.finetune.download.hf import export_dataset, load_dataset_split


def _load_prepare_fn(dataset_name: str) -> Callable[[Dataset], Dataset] | None:
    """Dynamically load prepare function from dataset directory.

    Looks for prepare.py in the dataset's directory and returns the prepare function.
    Returns None if no prepare.py exists.
    """
    prepare_path = Path(__file__).parent / dataset_name / "prepare.py"
    if not prepare_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(f"{dataset_name}.prepare", prepare_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "prepare", None)


@dataclass
class DatasetConfig:
    """Configuration for a registered dataset.

    Attributes:
        repo_id: HuggingFace dataset repository ID
        train_split: Split name for training data (default: "train")
        eval_split: Optional split name for evaluation data
        data_dir: Subdirectory within the repo (e.g., "PAR4PC" for PANORAMA)
        data_files: Specific file(s) to load (e.g., "chemcotbench-cot/mol_edit/add.json")
        name: Dataset config/subset name (for datasets with multiple configs)
        export_format: Export format - "json", "jsonl", "csv", or "parquet"
    """

    repo_id: str
    train_split: str = "train"
    # TODO: eval_split is currently NOT used in training. To enable eval during training:
    #   1. prepare() should also export eval split to eval.<format>
    #   2. process_data.py prompt should handle eval data
    #   3. train.yaml should include eval_dataset, do_eval=true, eval_strategy
    #   4. ...
    eval_split: str | None = None
    data_dir: str | None = None
    data_files: str | list[str] | None = None
    name: str | None = None
    export_format: str = "parquet"


# Dataset registry: name -> DatasetConfig
# Note: Only train_split is configured to prevent test data leakage
DATASETS: dict[str, DatasetConfig] = {
    # PANORAMA - Patent Analysis Tasks (3 separate tasks)
    "panorama-par4pc": DatasetConfig(
        repo_id="LG-AI-Research/PANORAMA",
        data_dir="PAR4PC",  # Prior Art Retrieval for Patent Claims
        train_split="train",
        eval_split="validation",
    ),
    "panorama-noc4pc": DatasetConfig(
        repo_id="LG-AI-Research/PANORAMA",
        data_dir="NOC4PC",  # Notice of Compliance for Patent Claims
        train_split="train",
        eval_split="validation",
    ),
    "panorama-pi4pc": DatasetConfig(
        repo_id="LG-AI-Research/PANORAMA",
        data_dir="PI4PC",  # Patent Infringement for Patent Claims
        train_split="train",
        eval_split="validation",
    ),
    # DeepScaleR
    "deepscaler": DatasetConfig(
        repo_id="agentica-org/DeepScaleR-Preview-Dataset",
        train_split="train",
    ),
    # ChemCoT - Chemical Reasoning with Chain-of-Thought (4 separate tasks)
    # Paper: https://arxiv.org/abs/2505.21318
    # All data files are under chemcotbench-cot/ directory
    "chemcot-mol_und": DatasetConfig(
        repo_id="OpenMol/ChemCoTDataset",
        data_dir="chemcotbench-cot/mol_und",  # Molecular understanding: functional group counting, ring counting, scaffold extraction
        train_split="train",
    ),
    "chemcot-mol_edit": DatasetConfig(
        repo_id="OpenMol/ChemCoTDataset",
        data_dir="chemcotbench-cot/mol_edit",  # Molecular editing: add, delete, substitute functional groups
        train_split="train",
    ),
    "chemcot-mol_opt": DatasetConfig(
        repo_id="OpenMol/ChemCoTDataset",
        data_dir="chemcotbench-cot/mol_opt",  # Molecular optimization: LogP, solubility, QED, drug targets
        train_split="train",
    ),
    # chemcot-rxn: uses prepare.py for schema unification (rcr.json uses 'cot_result' instead of 'struct_cot')
    "chemcot-rxn": DatasetConfig(
        repo_id="OpenMol/ChemCoTDataset",
        data_files=[
            "chemcotbench-cot/rxn/fs_major_product.json",
            "chemcotbench-cot/rxn/fs_by_product.json",
            "chemcotbench-cot/rxn/rcr.json",
        ],
        train_split="train",
    ),
}


def prepare(name: str, force: bool = False) -> str:
    """Download and export dataset train split to local directory.

    Only exports the train split to prevent test data leakage.
    The exported file is saved as: datasets/<name>/train.<format>

    If a prepare.py exists in the dataset directory, its prepare() function
    will be applied to transform the data before export.

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
    out_file = out_dir / f"train.{config.export_format}"

    if not force and out_file.exists():
        return str(out_dir)

    # Load prepare_fn from dataset directory if exists
    prepare_fn = _load_prepare_fn(name)

    # Load only the train split (never load test to prevent leakage)
    ds = load_dataset_split(
        repo_id=config.repo_id,
        split=config.train_split,
        name=config.name,
        data_dir=config.data_dir,
        data_files=config.data_files,
        prepare_fn=prepare_fn,
    )

    # Export to local file
    export_dataset(ds, str(out_file), format=config.export_format)

    # Copy custom README if exists
    custom_readme = Path(__file__).parent / name / "README.md"
    if custom_readme.exists():
        shutil.copy(custom_readme, out_dir / "README.md")

    return str(out_dir)


def load_split(name: str, split: str | None = None, cache_dir: str | None = None):
    """Load a specific split from a registered dataset as Dataset object.

    If a prepare.py exists in the dataset directory, its prepare() function
    will be applied to transform the data.

    Args:
        name: Dataset name (must be registered in DATASETS)
        split: Split to load ("train" or "eval"). Defaults to train_split.
        cache_dir: Local cache directory (default: HF cache)

    Returns:
        datasets.Dataset object
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    config = DATASETS[name]

    # Determine effective split
    if split == "eval" and config.eval_split:
        effective_split = config.eval_split
    else:
        effective_split = split or config.train_split

    # Load prepare_fn from dataset directory if exists
    prepare_fn = _load_prepare_fn(name)

    return load_dataset_split(
        repo_id=config.repo_id,
        split=effective_split,
        name=config.name,
        data_dir=config.data_dir,
        data_files=config.data_files,
        cache_dir=cache_dir,
        prepare_fn=prepare_fn,
    )


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
        # Default: show available datasets and load example
        print(f"Available datasets: {list(DATASETS.keys())}")
        ds = load_split("panorama-par4pc")
        print(f"Loaded PANORAMA PAR4PC train split: {len(ds)} samples")
        print(f"Columns: {ds.column_names}")
