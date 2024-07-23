from pathlib import Path

from pydantic_settings import BaseSettings


class PropSetting(BaseSettings):
    class Config:
        env_prefix = "DM_"  # Use MODEL_CODER_ as prefix for environment variables
        protected_namespaces = ()  # Add 'model_' to the protected namespaces

    # TODO: model part is not finished yet
    model_scen: str = "rdagent.scenarios.data_mining.experiment.model_experiment.DMModelScenario"
    model_hypothesis_gen: str = "rdagent.scenarios.data_mining.proposal.model_proposal.DMModelHypothesisGen"
    model_hypothesis2experiment: str = "rdagent.scenarios.data_mining.proposal.model_proposal.DMModelHypothesis2Experiment"
    model_coder: str = "rdagent.scenarios.data_mining.developer.model_coder.DMModelCoSTEER"
    model_runner: str = "rdagent.scenarios.data_mining.developer.model_runner.DMModelRunner"
    model_summarizer: str = "rdagent.scenarios.data_mining.developer.feedback.DMModelHypothesisExperiment2Feedback"

    evolving_n: int = 10

    py_bin: str = "/usr/bin/python"


PROP_SETTING = PropSetting()
