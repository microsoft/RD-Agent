"""
LLM Fine-tuning Entry Point

Standard RDLoop entry point for LLM fine-tuning, consistent with data science implementation.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

import fire

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.loop import LLMFinetuneRDLoop
from rdagent.scenarios.finetune.utils import ensure_ft_assets_exist


def main(
    model: str | None = None,
    dataset: str | None = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    timeout: Optional[str] = None,
):
    """
    LLM fine-tuning entry point

    Parameters
    ----------
    dataset : str
        Dataset name for fine-tuning (e.g., 'shibing624/alpaca-zh')
    model : str, optional
        Model name for fine-tuning (e.g., 'Qwen/Qwen2.5-1.5B-Instruct').
        If not provided, the system will automatically select an optimal model
        during hypothesis generation based on hardware constraints and dataset characteristics.
    step_n : int, optional
        Number of steps to run; if None, runs indefinitely until completion or error
    loop_n : int, optional
        Number of loops to run; if None, runs indefinitely until completion or error
    timeout : str, optional
        Maximum duration for the entire process

    Examples:
    .. code-block:: bash
        # With specified model
        dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset shibing624/alpaca-zh --model Qwen/Qwen2.5-1.5B-Instruct

        # With automatic model selection
        dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset shibing624/alpaca-zh
    """
    if not dataset:
        raise Exception("Please specify dataset name using --dataset")

    # Validate FT_FILE_PATH environment variable (auto-filled by pydantic)
    if not FT_RD_SETTING.file_path:
        raise Exception("Please set FT_FILE_PATH environment variable")
    ft_root = Path(FT_RD_SETTING.file_path)
    if not ft_root.exists():
        raise Exception(f"FT_FILE_PATH does not exist: {ft_root}")

    # Ensure dataset assets exist, model will be handled later if not specified
    ensure_ft_assets_exist(model, dataset)

    # Update FT setting instance with provided dataset and model (if specified)
    FT_RD_SETTING.dataset = dataset
    FT_RD_SETTING.base_model = model  # This can be None for auto-selection

    # Create and run LLM fine-tuning loop using standard RDLoop async workflow
    if model:
        logger.info(f"Starting LLM fine-tuning: {model} on {dataset}")
    else:
        logger.info(f"Starting LLM fine-tuning with auto-selected model on {dataset}")
    loop = LLMFinetuneRDLoop(FT_RD_SETTING)
    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=timeout))
    logger.info("LLM fine-tuning completed!")


if __name__ == "__main__":
    fire.Fire(main)
