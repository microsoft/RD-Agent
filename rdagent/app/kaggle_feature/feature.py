import fire

from rdagent.app.kaggle_feature.conf import PROP_SETTING
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.developer import Developer
from rdagent.core.exception import FactorEmptyError
from rdagent.core.proposal import Hypothesis2Experiment, HypothesisExperiment2Feedback, HypothesisGen, Trace
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger

class FeatureRDLoop(RDLoop):
    def __init__(self, PROP_SETTING: BasePropSetting):
        with logger.tag("init"):
            scen: Scenario = import_class(PROP_SETTING.scen)(PROP_SETTING.competition)
            logger.log_object(scen, tag="scenario")

            self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)
            logger.log_object(self.hypothesis_gen, tag="hypothesis generator")

            self.hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()
            logger.log_object(self.hypothesis2experiment, tag="hypothesis2experiment")

            self.coder: Developer = import_class(PROP_SETTING.coder)(scen)
            logger.log_object(self.coder, tag="coder")
            self.runner: Developer = import_class(PROP_SETTING.runner)(scen)
            logger.log_object(self.runner, tag="runner")

            self.summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
            logger.log_object(self.summarizer, tag="summarizer")
            self.trace = Trace(scen=scen)
            super(RDLoop, self).__init__()

    skip_loop_error = (FactorEmptyError,)


def main(path=None, step_n=None, competition=None):
    """
    Auto R&D Evolving loop for feature engineering.

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/kaggle_feature/feature.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter
    
    """
    if competition:
        PROP_SETTING.competition = competition
    if path is None:
        model_loop = FeatureRDLoop(PROP_SETTING)
    else:
        model_loop = FeatureRDLoop.load(path)
    model_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)
