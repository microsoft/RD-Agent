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
    scen: str = "rdagent.scenarios.kaggle.experiment.scenario.KGScenario"
    """Scenario class for data mining model"""

    hypothesis_gen: str = "rdagent.scenarios.kaggle.proposal.proposal.KGHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.kaggle.proposal.proposal.KGHypothesis2Experiment"
    """Hypothesis to experiment class"""

    feature_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGFactorCoSTEER"
    """Feature Coder class"""

    model_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGModelCoSTEER"
    """Model Coder class"""

    feature_runner: str = "rdagent.scenarios.kaggle.developer.runner.KGFactorRunner"
    """Feature Runner class"""

    model_runner: str = "rdagent.scenarios.kaggle.developer.runner.KGModelRunner"
    """Model Runner class"""

    summarizer: str = "rdagent.scenarios.kaggle.developer.feedback.KGHypothesisExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""

    competition: str = ""


PROP_SETTING = PropSetting()
