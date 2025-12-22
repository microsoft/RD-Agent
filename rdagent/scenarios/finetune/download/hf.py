import os
import shutil
from pathlib import Path
from typing import Optional


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


def download_dataset(
    repo_id: str,
    out_dir: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Download HuggingFace dataset to a specified directory using snapshot_download.
    Preserves the original file structure from HuggingFace.

    Args:
        repo_id: HuggingFace dataset repository ID
        out_dir: Directory to save the dataset
        token: HuggingFace token for private datasets
        revision: Specific revision to download
        force: If True, re-download even if exists

    Returns:
        Path to the downloaded dataset directory
    """
    save_path = Path(out_dir)
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
        repo_type="dataset",
        local_dir=str(save_path),
        local_dir_use_symlinks=False,
        token=_get_hf_token(token),
        revision=revision,
    )
    return str(save_path)


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
