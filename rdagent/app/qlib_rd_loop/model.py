"""
Model workflow with session control
"""

import asyncio

import fire

from rdagent.app.qlib_rd_loop.conf import MODEL_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import ModelEmptyError


class ModelRDLoop(RDLoop):
    skip_loop_error = (ModelEmptyError,)


def main(
    path=None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
    base_features_path: str | None = None,
    **kwargs
):
    """
    Auto R&D Evolving loop for fintech models

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/model.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter

    """
    if path is None:
        model_loop = ModelRDLoop(MODEL_PROP_SETTING)
    else:
        model_loop = ModelRDLoop.load(path, checkout=checkout)
    model_loop._init_base_features(base_features_path)
    if "user_interaction_queues" in kwargs and kwargs["user_interaction_queues"] is not None:
        model_loop._set_interactor(*kwargs["user_interaction_queues"])
        model_loop._interact_init_params()
    asyncio.run(model_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
