"""
Factor workflow with session control
"""

import fire

from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import FactorEmptyError


class FactorRDLoop(RDLoop):
    skip_loop_error = (FactorEmptyError,)


def main(path=None, step_n=None):
    """
    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/factor_w_sc.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter

    """
    if path is None:
        model_loop = FactorRDLoop(FACTOR_PROP_SETTING)
    else:
        model_loop = FactorRDLoop.load(path)
    model_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)
