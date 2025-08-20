"""
LLM Fine-tuning Entry Point

Simple entry point for LLM fine-tuning, similar to data science loop structure.
Delegates business logic to scenarios.finetune module.
"""

import os
from pathlib import Path

import fire

from rdagent.app.finetune.llm.conf import FT_RD_SETTING, update_settings
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.loop import LLMFinetuneRDLoop
from rdagent.scenarios.finetune.utils import ensure_ft_assets_exist


def main(
    model: str | None = None,
    dataset: str | None = None,
):
    """
    LLM fine-tuning entry point

    Parameters
    ----------
    dataset : str
        Dataset name for fine-tuning (e.g., 'shibing624/alpaca-zh')
    model : str
        Model name for fine-tuning (e.g., 'Qwen/Qwen2.5-1.5B-Instruct')

    Example:
    .. code-block:: bash
        dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset shibing624/alpaca-zh --model Qwen/Qwen2.5-1.5B-Instruct
    """
    if not dataset:
        raise Exception("Please specify dataset name using --dataset")

    if not model:
        raise Exception("Please specify model name using --model")

    # Validate FT_FILE_PATH environment variable
    ft_root_str = os.environ.get("FT_FILE_PATH")
    if not ft_root_str:
        raise Exception("Please set FT_FILE_PATH environment variable")

    ft_root = Path(ft_root_str)
    if not ft_root.exists():
        raise Exception(f"FT_FILE_PATH does not exist: {ft_root}")

    # Ensure dataset and model assets exist
    ensure_ft_assets_exist(model, dataset, ft_root)

    # Update settings with provided dataset and model
    update_settings(dataset, model)

    # Create and run LLM fine-tuning loop
    logger.info(f"Starting LLM fine-tuning: {model} on {dataset}")
    loop = LLMFinetuneRDLoop(dataset, model, FT_RD_SETTING)
    loop.run()
    logger.info("LLM fine-tuning completed!")


if __name__ == "__main__":
    fire.Fire(main)
