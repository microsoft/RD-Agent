import os
import shutil
from pathlib import Path
from typing import Optional
from datasets import load_dataset


def _ensure_parent(path: Path) -> None:
    os.makedirs(path.parent, mode=0o777, exist_ok=True)


def _get_hf_token(token: Optional[str] = None) -> Optional[str]:
    """Get HuggingFace token from parameter or environment variables."""
    return (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def load_dataset_split(
    repo_id: str,
    split: str = "train",
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[str | list[str]] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
):
    """
    Load a specific split from HuggingFace dataset using datasets library.

    Args:
        repo_id: HuggingFace dataset repository ID
        split: Dataset split to load ("train", "validation", "test")
        name: Dataset config/subset name (if dataset has multiple configs)
        data_dir: Subdirectory within the dataset repo (e.g., "PAR4PC" for PANORAMA)
        data_files: Specific file(s) to load (e.g., "chemcotbench-cot/mol_edit/add.json")
        cache_dir: Local cache directory
        token: HuggingFace token for private datasets

    Returns:
        datasets.Dataset object
    """
    return load_dataset(
        repo_id,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        split=split,
        cache_dir=cache_dir,
        token=_get_hf_token(token),
    )


def export_dataset(
    dataset,
    output_path: str,
    format: str = "json",
) -> str:
    """
    Export a Dataset object to a local file.

    Args:
        dataset: datasets.Dataset object to export
        output_path: Path to save the exported file
        format: Export format - "json", "jsonl", "csv", or "parquet"

    Returns:
        Path to the exported file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        dataset.to_json(str(path))
    elif format == "jsonl":
        dataset.to_json(str(path), lines=True)
    elif format == "csv":
        dataset.to_csv(str(path))
    elif format == "parquet":
        dataset.to_parquet(str(path))
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json', 'jsonl', 'csv', or 'parquet'.")

    return str(path)


def download_model(
    repo_id: str,
    out_dir_root: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Download Hugging Face model to a subdirectory under the specified root: <out_dir_root>/<repo_id>
    Returns the actual download directory path as a string.
    """
    if out_dir_root:
        save_root = Path(out_dir_root)
    else:
        # Use FT_RD_SETTING for default root directory
        from rdagent.app.finetune.llm.conf import FT_RD_SETTING

        if not FT_RD_SETTING.file_path:
            raise ValueError("No out_dir_root specified and FT_FILE_PATH not set")
        save_root = Path(FT_RD_SETTING.file_path) / "model"

    save_path = save_root / repo_id
    _ensure_parent(save_path)

    if force and save_path.exists():
        shutil.rmtree(save_path)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise ImportError(
            "huggingface_hub is missing. Please install it first: pip install -U 'huggingface_hub[cli]'"
        ) from e

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(save_path),
        local_dir_use_symlinks=False,
        token=_get_hf_token(token),
        revision=revision,
    )
    return str(save_path)


if __name__ == "__main__":
    # Example usage
    ds = load_dataset_split("shibing624/alpaca-zh", split="train")
    print(f"Loaded dataset with {len(ds)} samples")
