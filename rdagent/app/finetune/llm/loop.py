"""
LLM Fine-tuning Entry Point

Standard RDLoop entry point for LLM fine-tuning, consistent with data science implementation.
"""

import asyncio
from typing import Optional

import fire

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.loop import LLMFinetuneRDLoop


def main(
    dataset: Optional[str] = None,
    model: Optional[str] = None,
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
        If not provided, auto-selects optimal model based on hardware and dataset.
    step_n : int, optional
        Number of steps to run; if None, runs indefinitely until completion or error
    loop_n : int, optional
        Number of loops to run; if None, runs indefinitely until completion or error
    timeout : str, optional
        Maximum duration for the entire process

    Examples:
    .. code-block:: bash
        dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset shibing624/alpaca-zh --model Qwen/Qwen2.5-1.5B-Instruct
        dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset shibing624/alpaca-zh    # TODO: not enabled yet
    """
    if not dataset:
        logger.error("Please specify dataset name using --dataset")
        return

    # Update configuration with provided parameters
    FT_RD_SETTING.dataset = dataset
    if model:
        FT_RD_SETTING.base_model = model

    # Create and run LLM fine-tuning loop
    logger.info(f"Starting LLM fine-tuning on {dataset}" + (f" with {model}" if model else " (auto-select model)"))
    loop = LLMFinetuneRDLoop(FT_RD_SETTING)
    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=timeout))


if __name__ == "__main__":
    fire.Fire(main)
