from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting


class ModelBasePropSetting(BasePropSetting):
    model_config = SettingsConfigDict(env_prefix="QLIB_MODEL_", protected_namespaces=())

    # 1) override base settings
    scen: str = "rdagent.scenarios.qlib.experiment.model_experiment.QlibModelScenario"
    """Scenario class for Qlib Model"""

    hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.qlib.developer.model_coder.QlibModelCoSTEER"
    """Coder class"""

    runner: str = "rdagent.scenarios.qlib.developer.model_runner.QlibModelRunner"
    """Runner class"""

    summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibModelExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""


class FactorBasePropSetting(BasePropSetting):
    model_config = SettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    # 1) override base settings
    scen: str = "rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario"
    """Scenario class for Qlib Factor"""

    hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    """Coder class"""

    runner: str = "rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
    """Runner class"""

    summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibFactorExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""


class FactorFromReportPropSetting(FactorBasePropSetting):
    # 1) override the scen attribute
    scen: str = "rdagent.scenarios.qlib.experiment.factor_from_report_experiment.QlibFactorFromReportScenario"
    """Scenario class for Qlib Factor from Report"""

    # 2) sub task specific:
    report_result_json_file_path: str = "git_ignore_folder/report_list.json"
    """Path to the JSON file listing research reports for factor extraction"""

    max_factors_per_exp: int = 10000
    """Maximum number of factors implemented per experiment"""

    is_report_limit_enabled: bool = False
    """Limits report processing count if True; processes all if False"""


class QuantBasePropSetting(BasePropSetting):
    model_config = SettingsConfigDict(env_prefix="QLIB_QUANT_", protected_namespaces=())

    # 1) override base settings
    scen: str = "rdagent.scenarios.qlib.experiment.quant_experiment.QlibQuantScenario"
    """Scenario class for Qlib Model"""

    quant_hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.quant_proposal.QlibQuantHypothesisGen"
    """Hypothesis generation class"""

    model_hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesis2Experiment"
    """Hypothesis to experiment class"""

    model_coder: str = "rdagent.scenarios.qlib.developer.model_coder.QlibModelCoSTEER"
    """Coder class"""

    model_runner: str = "rdagent.scenarios.qlib.developer.model_runner.QlibModelRunner"
    """Runner class"""

    model_summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibModelExperiment2Feedback"
    """Summarizer class"""

    factor_hypothesis2experiment: str = (
        "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesis2Experiment"
    )
    """Hypothesis to experiment class"""

    factor_coder: str = "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    """Coder class"""

    factor_runner: str = "rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
    """Runner class"""

    factor_summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibFactorExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""

    action_selection: str = "bandit"
    """Action selection strategy: 'bandit' for bandit-based selection, 'llm' for LLM-based selection, 'random' for random selection"""


class QlibRunnerSettings(BasePropSetting):
    """Qlib runner settings"""

    model_config = SettingsConfigDict(env_prefix="QLIB_RUNNER_", protected_namespaces=())

    env_type: str = "conda"  # or "docker"
    """Environment to run the runner: 'conda' for local conda env, 'docker' for Docker container"""


FACTOR_PROP_SETTING = FactorBasePropSetting()
FACTOR_FROM_REPORT_PROP_SETTING = FactorFromReportPropSetting()
MODEL_PROP_SETTING = ModelBasePropSetting()
QUANT_PROP_SETTING = QuantBasePropSetting()
QLIB_RUNNER_SETTINGS = QlibRunnerSettings()
