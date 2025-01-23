from rdagent.app.kaggle.conf import KaggleBasePropSetting
from rdagent.core.conf import ExtendedSettingsConfigDict


class DataScienceBasePropSetting(KaggleBasePropSetting):
    model_config = ExtendedSettingsConfigDict(env_prefix="DS_", protected_namespaces=())

    # Main components
    ## Scen
    scen: str = "rdagent.scenarios.data_science.scen.KaggleScen"
    """Scenario class for data mining model"""

    ## proposal
    exp_gen: str = "rdagent.scenarios.data_science.proposal.exp_gen.DSExpGen"
    # exp_gen_init_kwargs: dict = {"max_trace_hist": 3}   # TODO: to be configurable

    # the two below should be used in ExpGen
    # hypothesis_gen: str = "rdagent.scenarios.kaggle.proposal.proposal.KGHypothesisGen"
    # """Hypothesis generation class"""
    #
    # hypothesis2experiment: str = "rdagent.scenarios.kaggle.proposal.proposal.KGHypothesis2Experiment"
    # """Hypothesis to experiment class"""

    ## dev/coder
    data_loader_coder: str = "rdagent.components.coder.data_science.raw_data_loader.DataLoaderCoSTEER"
    """Data Loader CoSTEER"""

    # feature_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGFactorCoSTEER"
    # """Feature Coder class"""

    # model_feature_selection_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGModelFeatureSelectionCoder"
    # """Model Feature Selection Coder class"""

    # model_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGModelCoSTEER"
    # """Model Coder class"""

    ## dev/runner
    feature_runner: str = "rdagent.scenarios.kaggle.developer.runner.KGFactorRunner"
    """Feature Runner class"""

    model_runner: str = "rdagent.scenarios.kaggle.developer.runner.KGModelRunner"
    """Model Runner class"""

    ## feedback
    summarizer: str = "rdagent.scenarios.kaggle.developer.feedback.KGExperiment2Feedback"
    """Summarizer class"""

    consecutive_errors: int = 5


DS_RD_SETTING = DataScienceBasePropSetting()
