from pathlib import Path

from pydantic_settings import BaseSettings


class PropSetting(BaseSettings):
    class Config:
        env_prefix = "QLIB_"  # Use MODEL_CODER_ as prefix for environment variables
        protected_namespaces = ()  # Add 'model_' to the protected namespaces

    factor_scen: str = "rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario"
    factor_hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesisGen"
    factor_hypothesis2experiment: str = (
        "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesis2Experiment"
    )
    factor_coder: str = "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    factor_runner: str = "rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
    factor_summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibFactorHypothesisExperiment2Feedback"

    # TODO: model part is not finished yet
    model_scen: str = "rdagent.scenarios.qlib.experiment.model_experiment.QlibModelScenario"
    model_hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesisGen"
    model_hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesis2Experiment"
    model_coder: str = "rdagent.scenarios.qlib.developer.model_coder.QlibModelCoSTEER"
    model_runner: str = "rdagent.scenarios.qlib.developer.model_runner.QlibModelRunner"
    model_summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibModelHypothesisExperiment2Feedback"

    evolving_n: int = 10

    py_bin: str = "/usr/bin/python"
    local_qlib_folder: Path = Path("/home/rdagent/qlib")


PROP_SETTING = PropSetting()
