"""
LLM Fine-tuning Entry Point

Standard RDLoop entry point for LLM fine-tuning, consistent with data science implementation.
"""

import asyncio
from typing import Optional, cast

import typer
from typing_extensions import Annotated

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.loop import LLMFinetuneRDLoop


def main(
    path: Optional[str] = None,
    checkout: Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")] = True,
    user_target_scenario: Optional[str] = None,
    benchmark: Optional[str] = None,
    benchmark_description: Optional[str] = None,
    dataset: Optional[str] = None,
    base_model: Optional[str] = None,
    upper_data_size_limit: Optional[int] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    timeout: Optional[str] = None,
):
    """
    LLM fine-tuning entry point

    Parameters
    ----------
    path :
        A path like `$LOG_PATH/__session__/1/0_propose`. This indicates that we restore the state after finishing step 0 in loop 1.
    checkout :
        Used to control the log session path. Boolean type, default is True.
        - If True, the new loop will use the existing folder and clear logs for sessions after the one corresponding to the given path.
        - If False, the new loop will use the existing folder but keep the logs for sessions after the one corresponding to the given path.
    dataset : str
        Dataset name for fine-tuning (e.g., 'shibing624/alpaca-zh')
    base_model : str, optional
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
        dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset shibing624/alpaca-zh --base-model Qwen/Qwen2.5-1.5B-Instruct
        dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset shibing624/alpaca-zh    # TODO: not enabled yet
    """

    if user_target_scenario:
        FT_RD_SETTING.user_target_scenario = user_target_scenario
    assert (
        FT_RD_SETTING.user_target_scenario is None
    ), "user_target_scenario is not yet supported, please specify via benchmark and benchmark_description"
    if upper_data_size_limit:
        FT_RD_SETTING.upper_data_size_limit = upper_data_size_limit
        logger.info(f"Set upper_data_size_limit to {FT_RD_SETTING.upper_data_size_limit}")
    if benchmark and benchmark_description:
        FT_RD_SETTING.target_benchmark = benchmark
        FT_RD_SETTING.benchmark_description = benchmark_description
    assert FT_RD_SETTING.user_target_scenario or (
        FT_RD_SETTING.target_benchmark and FT_RD_SETTING.benchmark_description
    ), "Either user_target_scenario or target_benchmark must be specified for LLM fine-tuning."

    # Update configuration with provided parameters
    if dataset:
        FT_RD_SETTING.dataset = dataset
    if base_model:
        FT_RD_SETTING.base_model = base_model

    # Create and run LLM fine-tuning loop
    data_set_target = FT_RD_SETTING.dataset if FT_RD_SETTING.dataset else "auto generated dataset"
    model_target = FT_RD_SETTING.base_model if FT_RD_SETTING.base_model else "auto selected model"

    # Temporary assertion until auto-selection is implemented
    assert (
        FT_RD_SETTING.base_model is not None
    ), "Base model auto selection not yet supported, please specify via --base-model"

    logger.info(f"Starting LLM fine-tuning on dataset='{data_set_target}' with model='{model_target}'")

    if path is None:
        loop = LLMFinetuneRDLoop(FT_RD_SETTING)
    else:
        loop = cast(LLMFinetuneRDLoop, LLMFinetuneRDLoop.load(str(path), checkout=checkout))

    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=timeout))


if __name__ == "__main__":
    typer.run(main)
