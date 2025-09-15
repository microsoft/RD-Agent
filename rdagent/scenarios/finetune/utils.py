from pathlib import Path

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.download import download_dataset, download_model


def ensure_ft_assets_exist(
    *, model: str | None = None, dataset: str | None = None, check_model: bool = False, check_dataset: bool = False
) -> None:
    """Ensure dataset and model assets exist under FT_FILE_PATH structure.

    Args:
        model: Model name to check/download. Required if check_model=True.
        dataset: Dataset name to check/download. Required if check_dataset=True.
        check_model: Whether to ensure model exists.
        check_dataset: Whether to ensure dataset exists.

    Paths:
        - Dataset path: FT_RD_SETTING.file_path/datasets/<dataset>
        - Model path:   FT_RD_SETTING.file_path/models/<model>
    """
    # Ensure dataset exists if requested
    if check_dataset:
        if dataset is None:
            raise ValueError("Dataset name is required when check_dataset=True")

        dataset_dir = Path(FT_RD_SETTING.file_path) / "datasets" / dataset
        if not dataset_dir.exists():
            try:
                logger.info(f"Downloading dataset '{dataset}' to {dataset_dir}")
                download_dataset(dataset, out_dir_root=str(Path(FT_RD_SETTING.file_path) / "datasets"))
            except Exception as e:
                raise Exception(f"Failed to download dataset '{dataset}' to {dataset_dir}: {e}") from e

    # Ensure model exists if requested
    if check_model:
        if model is None:
            raise ValueError("Model name is required when check_model=True")

        model_dir = Path(FT_RD_SETTING.file_path) / "models" / model
        if not model_dir.exists():
            try:
                logger.info(f"Downloading model '{model}' to {model_dir}")
                download_model(model, out_dir_root=str(Path(FT_RD_SETTING.file_path) / "models"))
            except Exception as e:
                raise Exception(f"Failed to download model '{model}' to {model_dir}: {e}. ") from e
