"""
RL Post-training Entry Point
"""

import asyncio
from typing import Optional

import typer
from typing_extensions import Annotated

from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.loop import RLPostTrainingRDLoop


def main(
    base_model: Annotated[Optional[str], typer.Option("--base-model", "-m")] = None,
    benchmark: Annotated[Optional[str], typer.Option("--benchmark", "-b")] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    timeout: Optional[str] = None,
):
    """
    RL post-training entry point

    Parameters
    ----------
    base_model : str
        Model name (e.g., 'Qwen2.5-Coder-0.5B-Instruct')
        Docker path: /models/{base_model}
    benchmark : str
        Benchmark/dataset name (e.g., 'gsm8k')
        Docker path: /data/{benchmark}
    step_n : int, optional
        Number of steps to run; if None, runs all steps per loop
    loop_n : int, optional
        Number of loops to run; if None, runs indefinitely
    timeout : str, optional
        Maximum duration for the entire process

    Examples
    --------
    .. code-block:: bash

        export RL_MODELS_DIR=/path/to/models
        export RL_DATA_DIR=/path/to/data
        python rdagent/app/rl/loop.py --base-model Qwen2.5-Coder-0.5B-Instruct --benchmark gsm8k
    """
    # Update config from CLI
    if base_model:
        RL_RD_SETTING.base_model = base_model
    if benchmark:
        RL_RD_SETTING.benchmark = benchmark

    logger.info(f"Starting RL post-training: model={RL_RD_SETTING.base_model}, benchmark={RL_RD_SETTING.benchmark}")

    # RDLoop 会自动根据 RL_RD_SETTING.scen 创建 Scenario
    # Scenario.__init__() 中会自动运行 baseline 评测
    loop = RLPostTrainingRDLoop(RL_RD_SETTING)
    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=timeout))


if __name__ == "__main__":
    typer.run(main)
