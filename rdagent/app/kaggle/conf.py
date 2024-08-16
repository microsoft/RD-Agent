from pathlib import Path

from pydantic_settings import BaseSettings

from rdagent.components.workflow.conf import BasePropSetting


class PropSetting(BasePropSetting):
    class Config:
        env_prefix = "KG_"
        """Use `KG_` as prefix for environment variables"""
        protected_namespaces = ()
        """Add 'model_' to the protected namespaces"""

    # 1) overriding the default
    scen: str = "rdagent.scenarios.kaggle.experiment.model_experiment.KGModelScenario"
    """Scenario class for data mining model"""

    hypothesis_gen: str = "rdagent.scenarios.kaggle.proposal.model_proposal.KGModelHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.kaggle.proposal.model_proposal.KGModelHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.kaggle.developer.model_coder.KGModelCoSTEER"
    """Coder class"""

    runner: str = "rdagent.scenarios.kaggle.developer.model_runner.KGModelRunner"
    """Runner class"""

    summarizer: str = "rdagent.scenarios.kaggle.developer.feedback.KGModelHypothesisExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""

    evolving_n: int = 10

    competition: str = ""


PROP_SETTING = PropSetting()
