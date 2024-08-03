from rdagent.components.workflow.conf import BasePropSetting


class ModelBasePropSetting(BasePropSetting):
    class Config:
        env_prefix = "QLIB_MODEL_"  # Use MODEL_CODER_ as prefix for environment variables
        protected_namespaces = ()  # Add 'model_' to the protected namespaces

    scen: str = "rdagent.scenarios.qlib.experiment.model_experiment.QlibModelScenario"
    hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesisGen"
    hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesis2Experiment"
    coder: str = "rdagent.scenarios.qlib.developer.model_coder.QlibModelCoSTEER"
    runner: str = "rdagent.scenarios.qlib.developer.model_runner.QlibModelRunner"
    summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibModelHypothesisExperiment2Feedback"

    evolving_n: int = 10


class FactorBasePropSetting(BasePropSetting):
    class Config:
        env_prefix = "QLIB_FACTOR_"  # Use MODEL_CODER_ as prefix for environment variables
        protected_namespaces = ()  # Add 'model_' to the protected namespaces

    # 1) override base settings
    # TODO: model part is not finished yet
    scen: str = "rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario"
    hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesisGen"
    hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesis2Experiment"
    coder: str = "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    runner: str = "rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
    summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibFactorHypothesisExperiment2Feedback"

    evolving_n: int = 10

    # 2) sub task specific:
    origin_report_path: str = "data/report_origin"
    local_report_path: str = "data/report"
    report_result_json_file_path: str = "git_ignore_folder/report_list.json"
    progress_file_path: str = "git_ignore_folder/progress.pkl"
    report_extract_result: str = "git_ignore_folder/hypo_exp_cache.pkl"
    max_factor_per_report: int = 10000


FACTOR_PROP_SETTING = FactorBasePropSetting()
MODEL_PROP_SETTING = ModelBasePropSetting()
