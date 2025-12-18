"""Dataset preparation module for finetune scenarios.

Usage:
    from rdagent.scenarios.finetune.datasets import prepare, prepare_all

    prepare("deepscaler")   # Prepare single dataset
    prepare("chemcot")      # Prepare ChemCoT dataset (with JSON consolidation)
    prepare_all()           # Prepare all registered datasets
"""

import json
import shutil
from pathlib import Path

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.scenarios.finetune.download.hf import download_dataset

# Dataset registry: name -> HuggingFace repo_id
# Empty for debug: use local limo dataset instead of downloading
DATASETS = {
    "deepscaler": "agentica-org/DeepScaleR-Preview-Dataset",  # Removed for debug(Need to prepare data manually)
    "chemcot": "OpenMol/ChemCoTDataset",
}

# Datasets that require post-processing (consolidation)
DATASETS_WITH_POSTPROCESS = {"chemcot"}


def _consolidate_chemcot_jsons(dataset_path: Path) -> None:
    """Consolidate all ChemCoT JSON files into a single JSON file.

    The consolidated file will have the structure:
    [
        {
            "id": "...",
            "query": "...",
            "task": "rxn|mol_und|mol_edit|mol_opt",
            "subtask": "...",
            "struct_cot": "...",
            "raw_cot": "...",
            "meta": "..."
        },
        ...
    ]

    After consolidation, the original downloaded subdirectories are removed,
    keeping only consolidated.json and README.md.

    Args:
        dataset_path: Path to the downloaded ChemCoT dataset
    """
    consolidated_data = []
    subdirs = ["rxn", "mol_und", "mol_edit", "mol_opt"]

    # Find the actual data root (may be nested in chemcotbench-cot/)
    data_root = dataset_path
    if (dataset_path / "chemcotbench-cot").exists():
        data_root = dataset_path / "chemcotbench-cot"

    for subdir in subdirs:
        subdir_path = data_root / subdir
        if not subdir_path.exists():
            continue

        for json_file in subdir_path.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    # Each item should already have task/subtask fields
                    consolidated_data.extend(data)
                else:
                    # Single object
                    consolidated_data.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue

    # Write consolidated JSON to dataset root
    output_file = dataset_path / "consolidated.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(consolidated_data, f, ensure_ascii=False, indent=2)

    print(f"Consolidated {len(consolidated_data)} samples into {output_file}")

    # Clean up: remove ALL other files (including hidden), keep only consolidated.json and README.md
    keep_files = {"consolidated.json", "README.md"}
    for item in dataset_path.iterdir():
        if item.name not in keep_files:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    print(f"Cleaned up original files, kept only: {', '.join(keep_files)}")


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

    # Post-processing for specific datasets
    if name in DATASETS_WITH_POSTPROCESS:
        if name == "chemcot":
            _consolidate_chemcot_jsons(save_path)

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
    import sys

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "deepscaler"
    path = prepare(dataset_name)
    print(f"Dataset prepared at: {path}")
