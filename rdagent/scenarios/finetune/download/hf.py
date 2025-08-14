import os
from pathlib import Path
from typing import Optional


def _resolve_root_dir(preferred_root: Optional[str], fallback_env_keys: list[str]) -> Path:
    if preferred_root:
        return Path(preferred_root)
    for key in fallback_env_keys:
        val = os.environ.get(key)
        if val:
            return Path(val)
    raise ValueError(
        f"Root directory not specified. Please provide out_dir_root or set one of the environment variables: {', '.join(fallback_env_keys)}."
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_dataset(
    repo_id: str,
    out_dir_root: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Download Hugging Face dataset to a subdirectory under the specified root: <out_dir_root>/<repo_id>
    Returns the actual download directory path as a string.
    """
    save_root = _resolve_root_dir(out_dir_root, ["FT_FILE_PATH", "DS_LOCAL_DATA_PATH"])
    # When using FT_FILE_PATH, datasets are usually placed in the 'dataset' subdirectory
    if os.environ.get("FT_FILE_PATH") and Path(os.environ["FT_FILE_PATH"]) == save_root:
        save_root = save_root / "dataset"
    save_path = save_root / repo_id
    _ensure_parent(save_path)

    if force and save_path.exists():
        import shutil

        shutil.rmtree(save_path)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise ImportError(
            "huggingface_hub is missing. Please install it first: pip install -U 'huggingface_hub[cli]'"
        ) from e

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
    save_root = _resolve_root_dir(out_dir_root, ["FT_FILE_PATH", "HF_MODEL_PATH", "MODEL_DIR", "MODELS_DIR"])
    save_path = save_root / repo_id
    _ensure_parent(save_path)

    if force and save_path.exists():
        import shutil

        shutil.rmtree(save_path)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise ImportError(
            "huggingface_hub is missing. Please install it first: pip install -U 'huggingface_hub[cli]'"
        ) from e

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
    return str(save_path)
