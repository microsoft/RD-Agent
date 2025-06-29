"""
Factor workflow with session control
"""

import asyncio
from pathlib import Path
from typing import Any

import fire

from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger


class FactorRDLoop(RDLoop):
    skip_loop_error = (FactorEmptyError,)

    def running(self, prev_out: dict[str, Any]):
        exp = self.runner.develop(prev_out["coding"])
        if exp is None:
            logger.error(f"Factor extraction failed.")
            raise FactorEmptyError("Factor extraction failed.")
        logger.log_object(exp, tag="runner result")
        return exp


def main(
    path: str | None = None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool | str | Path = True,
):
    """
    Auto R&D Evolving loop for fintech factors.

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/factor.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter

    """
    if path is None:
        model_loop = FactorRDLoop(FACTOR_PROP_SETTING)
    else:
        model_loop = FactorRDLoop.load(path, checkout=checkout)
    asyncio.run(model_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
