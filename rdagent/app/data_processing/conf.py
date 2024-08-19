from pathlib import Path

from pydantic_settings import BaseSettings

from rdagent.components.workflow.conf import BasePropSetting


class PropSetting(BasePropSetting):
    class Config:
        env_prefix = "DP_"
        """Use `DP_` as prefix for environment variables"""
        protected_namespaces = ()
        """Add 'feature_' to the protected namespaces"""

    # 1) overriding the default
    scen: str = "rdagent.scenarios.data_processing.experiment.feature_experiment.DPFeatureScenario"
    """Scenario class for data processing model"""

    hypothesis_gen: str = "rdagent.scenarios.data_processing.proposal.factor_proposal.DPFeatureHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.data_processing.proposal.factor_proposal.DPFeatureHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.data_processing.developer.data_coder.DPFeatureCoSTEER"
    """Coder class"""

    runner: str = "rdagent.scenarios.data_processing.developer.data_runner.DPFeatureRunner"
    """Runner class"""

    summarizer: str = "rdagent.scenarios.data_processing.developer.feedback.DPFeatureHypothesisExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""

    evolving_n: int = 10

    # 2) Extra config for the scenario
    # physionet account
    # NOTE: You should apply the account in https://physionet.org/
    username: str = ""
    """Physionet account username"""

    password: str = ""
    """Physionet account password"""


PROP_SETTING = PropSetting()
