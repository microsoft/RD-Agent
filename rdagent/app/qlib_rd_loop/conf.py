from pydantic_settings import BaseSettings


class PropSetting(BaseSettings):
    """"""

    scen: str = "rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario"
    hypothesis_gen: str = "rdagent.scenarios.qlib.factor_proposal.QlibFactorHypothesisGen"
    hypothesis2experiment: str = "rdagent.scenarios.qlib.factor_proposal.QlibFactorHypothesis2Experiment"
    qlib_factor_coder: str = "rdagent.components.task_implementation.factor_implementation.CoSTEER.CoSTEERFG"
    qlib_factor_runner: str = "rdagent.scenarios.qlib.task_generator.data.QlibFactorRunner"
    qlib_factor_summarizer: str = "rdagent.scenarios.qlib.task_generator.feedback.QlibFactorExperiment2Feedback"

    evolving_n: int = 10


PROP_SETTING = PropSetting()
