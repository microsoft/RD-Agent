import os
from pathlib import Path
from typing import Optional


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
    if out_dir_root:
        save_root = Path(out_dir_root)
    else:
        # Use FT_RD_SETTING for default root directory
        from rdagent.app.finetune.llm.conf import FT_RD_SETTING

        if not FT_RD_SETTING.file_path:
            raise ValueError("No out_dir_root specified and FT_FILE_PATH not set")
        save_root = Path(FT_RD_SETTING.file_path) / "dataset"

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
