from pathlib import Path

from rdagent.components.workflow.conf import BasePropSetting


class PropSetting(BasePropSetting):
    class Config:
        env_prefix = "DM_"  # Use MODEL_CODER_ as prefix for environment variables
        protected_namespaces = ()  # Add 'model_' to the protected namespaces

    # 1) overriding the default
    scen: str = "rdagent.scenarios.data_mining.experiment.model_experiment.DMModelScenario"
    hypothesis_gen: str = "rdagent.scenarios.data_mining.proposal.model_proposal.DMModelHypothesisGen"
    hypothesis2experiment: str = "rdagent.scenarios.data_mining.proposal.model_proposal.DMModelHypothesis2Experiment"
    coder: str = "rdagent.scenarios.data_mining.developer.model_coder.DMModelCoSTEER"
    runner: str = "rdagent.scenarios.data_mining.developer.model_runner.DMModelRunner"
    summarizer: str = "rdagent.scenarios.data_mining.developer.feedback.DMModelHypothesisExperiment2Feedback"

    evolving_n: int = 10

    # 2) Extra config for the scenario
    # physionet account
    # NOTE: You should apply the account in https://physionet.org/
    username: str = ""
    password: str = ""


PROP_SETTING = PropSetting()
