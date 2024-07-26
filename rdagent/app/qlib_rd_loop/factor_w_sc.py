"""
Factor workflow with session control
"""

from typing import Any

import fire

from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger


class FactorRDLoop(RDLoop):
    skip_loop_error = (FactorEmptyError,)

    def exp_gen(self, prev_out: dict[str, Any]):
        with logger.tag("r"):  # research
            exp = self.hypothesis2experiment.convert(prev_out["propose"], self.trace)
            if exp is None:
                logger.error(f"Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")
            logger.log_object(exp.sub_tasks, tag="experiment generation")
        return exp


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
