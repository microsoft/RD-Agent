from pathlib import Path

from pydantic_settings import BaseSettings

from rdagent.components.workflow.conf import BasePropSetting


class PropSetting(BasePropSetting):
    class Config:
        env_prefix = "KGF_"
        """Use `KGF_` as prefix for environment variables"""
        protected_namespaces = ()
        """Add 'model_' to the protected namespaces"""

    # 1) overriding the default
    scen: str = "rdagent.scenarios.kaggle_feature.experiment.feature_experiment.KGFeatureScenario"
    """Scenario class for data mining model"""

    hypothesis_gen: str = "rdagent.scenarios.kaggle_feature.proposal.feature_proposal.KGFeatureHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.kaggle_feature.proposal.feature_proposal.KGFeatureHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.kaggle_feature.developer.feature_coder.KGFeatureCoSTEER"
    """Coder class"""

    runner: str = "rdagent.scenarios.kaggle_feature.developer.feature_runner.KGFeatureRunner"
    """Runner class"""

    summarizer: str = "rdagent.scenarios.kaggle_feature.developer.feedback.KGFeatureHypothesisExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""

    competition: str = ""


PROP_SETTING = PropSetting()
