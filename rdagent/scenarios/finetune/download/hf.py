import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

from rdagent.log import rdagent_logger as logger


def download_dataset(
    repo_id: str,
    out_dir_root: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """
    Download Hugging Face dataset to a subdirectory under the specified root: <out_dir_root>/<repo_id>

    Args:
        repo_id: HuggingFace dataset repository ID (e.g., "username/dataset-name")
        out_dir_root: Root directory for downloads. If None, raises ValueError.
        token: HuggingFace API token (optional, will try environment variables)
        revision: Specific revision/branch to download (optional)

    Returns:
        Path to the downloaded dataset directory as a string.
    """
    if not out_dir_root:
        raise ValueError("out_dir_root must be specified for dataset downloads")

    save_root = Path(out_dir_root)
    # Ensure root directory exists
    save_root.mkdir(parents=True, exist_ok=True)

    save_path = save_root / repo_id

    # Check if dataset already exists, skip download
    if save_path.exists():
        logger.info(f"Dataset already exists at {save_path}, skipping download")
        return str(save_path)

    logger.info(f"Downloading dataset {repo_id} to {save_path}")
    effective_token = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(save_path),
        local_dir_use_symlinks=False,
        token=effective_token,
        revision=revision,
    )
    logger.info(f"Dataset {repo_id} downloaded successfully to {save_path}")
    return str(save_path)


def download_model(
    repo_id: str,
    out_dir_root: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """
    Download Hugging Face model to a subdirectory under the specified root: <out_dir_root>/<repo_id>

    Args:
        repo_id: HuggingFace model repository ID (e.g., "username/model-name")
        out_dir_root: Root directory for downloads. If None, raises ValueError.
        token: HuggingFace API token (optional, will try environment variables)
        revision: Specific revision/branch to download (optional)

    Returns:
        Path to the downloaded model directory as a string.
    """
    if not out_dir_root:
        raise ValueError("out_dir_root must be specified for model downloads")

    save_root = Path(out_dir_root)
    # Ensure root directory exists
    save_root.mkdir(parents=True, exist_ok=True)

    save_path = save_root / repo_id

    # Check if model already exists, skip download
    if save_path.exists():
        logger.info(f"Model already exists at {save_path}, skipping download")
        return str(save_path)

    logger.info(f"Downloading model {repo_id} to {save_path}")
    effective_token = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(save_path),
        local_dir_use_symlinks=False,
        token=effective_token,
        revision=revision,
    )
    logger.info(f"Model {repo_id} downloaded successfully to {save_path}")
    return str(save_path)
