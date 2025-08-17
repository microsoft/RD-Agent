from pathlib import Path

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.download import download_dataset, download_model


def safe_path_component(s: str) -> str:
    """Convert a string to a safe path component by replacing path separators with '__'."""
    return s.replace("/", "__").replace("\\", "__")


def prev_model_dirname(model: str, dataset: str) -> str:
    """Generate prev_model directory name using safe model and dataset names."""
    return f"{safe_path_component(model)}_{safe_path_component(dataset)}"


def ensure_ft_assets_exist(model: str | None, dataset: str, ft_root: Path) -> None:
    """Ensure dataset and model assets exist under FT_FILE_PATH structure.

    - Dataset path: <ft_root>/dataset/<dataset>
    - Model path:   <ft_root>/model/<model>
    - Prev path:    <ft_root>/prev_model/<model>_<dataset>
    """
    dataset_dir = ft_root / "dataset" / dataset
    if not dataset_dir.exists():
        try:
            logger.info(f"Downloading dataset '{dataset}' to {dataset_dir}")
            download_dataset(dataset, out_dir_root=str(ft_root / "dataset"))
        except Exception as e:
            raise Exception(f"Failed to download dataset '{dataset}' to {dataset_dir}: {e}") from e

    # Model may be optional for some flows, but for finetune we typically require one of prev_model or model
    if model is not None:
        prev_dir = ft_root / "prev_model" / prev_model_dirname(model, dataset)
        model_dir = ft_root / "model" / model
        if not prev_dir.exists() and not model_dir.exists():
            try:
                logger.info(f"Downloading model '{model}' to {model_dir}")
                download_model(model, out_dir_root=str(ft_root / "model"))
            except Exception as e:
                raise Exception(
                    f"Failed to download model '{model}' to {model_dir}: {e}. "
                    f"At least one of prev_model or model is required.",
                ) from e
