import fire

from rdagent.app.data_mining.conf import MED_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import ModelEmptyError


class ModelRDLoop(RDLoop):
    skip_loop_error = (ModelEmptyError,)


def main(path=None, step_n=None):
    """
    Auto R&D Evolving loop for models in a medical scenario.

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/data_mining/model.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter

    """
    if path is None:
        model_loop = ModelRDLoop(MED_PROP_SETTING)
    else:
        model_loop = ModelRDLoop.load(path)
    model_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)
