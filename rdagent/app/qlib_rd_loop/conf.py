from pydantic_settings import BaseSettings


class PropSetting(BaseSettings):
    """"""

    qlib_factor_scen: str = "rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario"
    qlib_factor_hypothesis_gen: str = "rdagent.scenarios.qlib.factor_proposal.QlibFactorHypothesisGen"
    qlib_factor_hypothesis2experiment: str = "rdagent.scenarios.qlib.factor_proposal.QlibFactorHypothesis2Experiment"
    qlib_factor_coder: str = "rdagent.scenarios.qlib.factor_task_implementation.QlibFactorCoSTEER"
    qlib_factor_runner: str = "rdagent.scenarios.qlib.task_generator.data.QlibFactorRunner"
    qlib_factor_summarizer: str = (
        "rdagent.scenarios.qlib.task_generator.feedback.QlibFactorHypothesisExperiment2Feedback"
    )

    # TODO: model part is not finished yet
    qlib_model_scen: str = ""
    qlib_model_hypothesis_gen: str = ""
    qlib_model_hypothesis2experiment: str = ""
    qlib_model_coder: str = ""
    qlib_model_runner: str = ""
    qlib_model_summarizer: str = ""

    evolving_n: int = 10


PROP_SETTING = PropSetting()
