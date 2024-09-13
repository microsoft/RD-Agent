from pathlib import Path

from pydantic_settings import BaseSettings

from rdagent.components.workflow.conf import BasePropSetting


class MedBasePropSetting(BasePropSetting):
    class Config:
        env_prefix = "DM_"
        """Use `DM_` as prefix for environment variables"""
        protected_namespaces = ()
        """Add 'model_' to the protected namespaces"""

    # 1) overriding the default
    scen: str = "rdagent.scenarios.data_mining.experiment.model_experiment.DMModelScenario"
    """Scenario class for data mining model"""

    hypothesis_gen: str = "rdagent.scenarios.data_mining.proposal.model_proposal.DMModelHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.data_mining.proposal.model_proposal.DMModelHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.data_mining.developer.model_coder.DMModelCoSTEER"
    """Coder class"""

    runner: str = "rdagent.scenarios.data_mining.developer.model_runner.DMModelRunner"
    """Runner class"""

    summarizer: str = "rdagent.scenarios.data_mining.developer.feedback.DMModelHypothesisExperiment2Feedback"
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


MED_PROP_SETTING = MedBasePropSetting()
