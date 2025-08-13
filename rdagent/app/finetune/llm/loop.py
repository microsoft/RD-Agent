import asyncio
import os
from pathlib import Path

import fire

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.app.finetune.llm.conf import update_settings
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.loop import DataScienceRDLoop


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
        dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset shibing624/alpaca-zh
    """
    if not dataset:
        raise Exception("Please specify dataset name.")

    ft_root_str = os.environ.get("FT_FILE_PATH")
    if not ft_root_str:
        raise Exception("Please set FT_FILE_PATH in your .env.")
    ft_root = Path(ft_root_str)
    if not ft_root.exists():
        raise Exception(f"FT_FILE_PATH does not exist: {ft_root}")
    prev_dir = ft_root / "prev_model" / dataset
    model_dir = ft_root / "model" / dataset
    dataset_dir = ft_root / "dataset" / dataset
    if not dataset_dir.exists():
        raise Exception(f"Dataset not found: {dataset_dir}")
    # Require at least one of prev_model or model to exist for finetune
    if not prev_dir.exists() and not model_dir.exists():
        raise Exception(
            f"Neither prev_model nor model exists for '{dataset}'. Please create one of: {prev_dir} or {model_dir}"
        )
    update_settings(dataset)
    rd_loop: DataScienceRDLoop = DataScienceRDLoop(DS_RD_SETTING)
    asyncio.run(rd_loop.run())


if __name__ == "__main__":
    fire.Fire(main)
