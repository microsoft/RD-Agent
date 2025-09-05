from pathlib import Path

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.download import download_dataset, download_model


def ensure_ft_assets_exist(model: str | None, dataset: str) -> None:
    """Ensure dataset and model assets exist under FT_FILE_PATH structure.

    - Dataset path: FT_RD_SETTING.file_path/datasets/<dataset>
    - Model path:   FT_RD_SETTING.file_path/models/<model>
    """
    # Import here to avoid circular imports
    dataset_dir = Path(FT_RD_SETTING.file_path) / "datasets" / dataset
    if not dataset_dir.exists():
        try:
            logger.info(f"Downloading dataset '{dataset}' to {dataset_dir}")
            download_dataset(dataset, out_dir_root=str(Path(FT_RD_SETTING.file_path) / "datasets"))
        except Exception as e:
            raise Exception(f"Failed to download dataset '{dataset}' to {dataset_dir}: {e}") from e

    # Model may be optional for some flows, but for finetune we typically require a model
    if model is not None:
        model_dir = Path(FT_RD_SETTING.file_path) / "models" / model
        if not model_dir.exists():
            try:
                logger.info(f"Downloading model '{model}' to {model_dir}")
                download_model(model, out_dir_root=str(Path(FT_RD_SETTING.file_path) / "models"))
            except Exception as e:
                raise Exception(f"Failed to download model '{model}' to {model_dir}: {e}. ") from e
