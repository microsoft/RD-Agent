import asyncio
import os
from pathlib import Path

import fire

from rdagent.app.finetune.llm.conf import FT_RD_SETTING, update_settings
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.download import download_dataset, download_model
from rdagent.scenarios.finetune.loop import FinetuneRDLoop
from rdagent.scenarios.finetune.utils import prev_model_dirname


def main(
    model: str | None = None,
    dataset: str | None = None,
):
    """
    Parameters
    ----------
    dataset :
        Dateset name, used for finetune.

    Auto R&D Evolving loop for models finetune.
    You can continue running a session by using the command:
    .. code-block:: bash
        dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset shibing624/alpaca-zh --model Qwen/Qwen2.5-1.5B-Instruct
    """
    if not dataset:
        raise Exception("Please specify dataset name.")

    ft_root_str = os.environ.get("FT_FILE_PATH")
    if not ft_root_str:
        raise Exception("Please set FT_FILE_PATH in your .env.")
    ft_root = Path(ft_root_str)
    if not ft_root.exists():
        raise Exception(f"FT_FILE_PATH does not exist: {ft_root}")
    ensure_ft_assets_exist(model, dataset, ft_root)
    update_settings(dataset, model)
    # Use LLM-specific setting instance instead of global DS_RD_SETTING
    rd_loop: FinetuneRDLoop = FinetuneRDLoop(FT_RD_SETTING)
    asyncio.run(rd_loop.run())


def ensure_ft_assets_exist(model: str | None, dataset: str, ft_root: Path) -> None:
    """Ensure dataset and model assets exist under FT_FILE_PATH structure.

    - Dataset path: <ft_root>/dataset/<dataset>
    - Model path:   <ft_root>/model/<model>
    - Prev path:    <ft_root>/prev_model/<model>_<dataset>
    """
    dataset_dir = ft_root / "dataset" / dataset
    if not dataset_dir.exists():
        try:
            download_dataset(dataset, out_dir_root=str(ft_root / "dataset"))
        except Exception as e:
            raise Exception(f"Failed to download dataset '{dataset}' to {dataset_dir}: {e}") from e

    # Model may be optional for some flows, but for finetune we typically require one of prev_model or model
    if model is not None:
        prev_dir = ft_root / "prev_model" / prev_model_dirname(model, dataset)
        model_dir = ft_root / "model" / model
        if not prev_dir.exists() and not model_dir.exists():
            try:
                download_model(model, out_dir_root=str(ft_root / "model"))
            except Exception as e:
                raise Exception(
                    f"Failed to download model '{model}' to {model_dir}: {e}. "
                    f"At least one of prev_model or model is required."
                ) from e


if __name__ == "__main__":
    fire.Fire(main)
