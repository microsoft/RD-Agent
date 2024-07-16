from pathlib import Path

from pydantic_settings import BaseSettings


class PropSetting(BaseSettings):
    """"""

    qlib_factor_scen: str = "rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario"
    qlib_factor_hypothesis_gen: str = "rdagent.scenarios.qlib.factor_proposal.QlibFactorHypothesisGen"
    qlib_factor_hypothesis2experiment: str = "rdagent.scenarios.qlib.factor_proposal.QlibFactorHypothesis2Experiment"
    qlib_factor_coder: str = "rdagent.scenarios.qlib.factor_coder.QlibFactorCoSTEER"
    qlib_factor_runner: str = "rdagent.scenarios.qlib.task_generator.data.QlibFactorRunner"
    qlib_factor_summarizer: str = (
        "rdagent.scenarios.qlib.task_generator.feedback.QlibFactorHypothesisExperiment2Feedback"
    )

    # TODO: model part is not finished yet
    qlib_model_scen: str = "rdagent.scenarios.qlib.experiment.model_experiment.QlibModelScenario"
    qlib_model_hypothesis_gen: str = "rdagent.scenarios.qlib.model_proposal.QlibModelHypothesisGen"
    qlib_model_hypothesis2experiment: str = "rdagent.scenarios.qlib.model_proposal.QlibModelHypothesis2Experiment"
    qlib_model_coder: str = "rdagent.scenarios.qlib.model_coder.QlibModelCoSTEER"
    qlib_model_runner: str = "rdagent.scenarios.qlib.task_generator.model.QlibModelRunner"
    qlib_model_summarizer: str = "rdagent.scenarios.qlib.task_generator.feedback.QlibModelHypothesisExperiment2Feedback"

    evolving_n: int = 10

    py_bin: str = "/usr/bin/python"
    local_qlib_folder: Path = Path("/home/rdagent/qlib")


PROP_SETTING = PropSetting()
