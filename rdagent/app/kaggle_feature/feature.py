import fire

from rdagent.app.kaggle_feature.conf import PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import FactorEmptyError


class FeatureRDLoop(RDLoop):
    skip_loop_error = (FactorEmptyError,)


def main(path=None, step_n=None):
    """
    Auto R&D Evolving loop for feature engineering.
    You can continue running session by
    .. code-block:: python
        dotenv run -- python rdagent/app/kaggle_feature/feature.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter
    """
    if path is None:
        model_loop = FeatureRDLoop(PROP_SETTING)
    else:
        model_loop = FeatureRDLoop.load(path)
    model_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)
