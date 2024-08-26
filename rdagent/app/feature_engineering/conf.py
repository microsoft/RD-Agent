from pathlib import Path

from pydantic_settings import BaseSettings

from rdagent.components.workflow.conf import BasePropSetting


class PropSetting(BasePropSetting):
    class Config:
        env_prefix = "FE_"
        """Use `FE_` as prefix for environment variables"""
        protected_namespaces = ()
        """Add 'feature_' to the protected namespaces"""

    # 1) overriding the default
    scen: str = "rdagent.scenarios.feature_engineering.experiment.feature_experiment.FEFeatureScenario"
    """Scenario class for data processing model"""

    hypothesis_gen: str = "rdagent.scenarios.feature_engineering.proposal.factor_proposal.FEFeatureHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.feature_engineering.proposal.factor_proposal.FEFeatureHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.feature_engineering.developer.data_coder.FEFeatureCoSTEER"
    """Coder class"""

    runner: str = "rdagent.scenarios.feature_engineering.developer.data_runner.FEFeatureRunner"
    """Runner class"""

    summarizer: str = "rdagent.scenarios.feature_engineering.developer.feedback.FEFeatureHypothesisExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""

    evolving_n: int = 10


PROP_SETTING = PropSetting()
