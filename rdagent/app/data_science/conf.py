from rdagent.components.workflow.conf import BasePropSetting
from rdagent.core.conf import ExtendedSettingsConfigDict


class KaggleBasePropSetting(BasePropSetting):
    model_config = ExtendedSettingsConfigDict(env_prefix="DS_", protected_namespaces=())

    # Main components
    ## Scen
    scen: str = "rdagent.scenarios.kaggle.experiment.scenario.KGScenario"
    """Scenario class for data mining model"""

    ## proposal
    hypothesis_gen: str = "rdagent.scenarios.kaggle.proposal.proposal.KGHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.kaggle.proposal.proposal.KGHypothesis2Experiment"
    """Hypothesis to experiment class"""

    ## dev/coder
    feature_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGFactorCoSTEER"
    """Feature Coder class"""

    model_feature_selection_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGModelFeatureSelectionCoder"
    """Model Feature Selection Coder class"""

    model_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGModelCoSTEER"
    """Model Coder class"""

    ## dev/runner
    feature_runner: str = "rdagent.scenarios.kaggle.developer.runner.KGFactorRunner"
    """Feature Runner class"""

    model_runner: str = "rdagent.scenarios.kaggle.developer.runner.KGModelRunner"
    """Model Runner class"""

    ## feedback
    summarizer: str = "rdagent.scenarios.kaggle.developer.feedback.KGExperiment2Feedback"
    """Summarizer class"""

    # Configs
    ## Base
    competition: str = ""
    """Kaggle competition name, e.g., 'sf-crime'"""

    template_path: str = "rdagent/scenarios/kaggle/experiment/templates"  # TODO: we may not need this
    """Kaggle competition base templates path"""

    local_data_path: str = ""
    """Folder storing Kaggle competition data"""

    if_using_mle_data: bool = False

    ## Workflow
    evolving_n: int = 10
    """Number of evolutions"""

    auto_submit: bool = False
    """Automatically upload and submit each experiment result to Kaggle platform"""

    ### shared components in the workflow
    # Conditionally set the knowledge_base based on the use of graph RAG
    knowledge_base: str = ""
    """Knowledge base class, uses 'KGKnowledgeGraph' when advanced graph-based RAG is enabled, otherwise empty."""

    domain_knowledge_path: str = "/data/userdata/share/kaggle/domain_knowledge"  # TODO: It should be sth like knowledge_base_kwargs
    """Folder storing domain knowledge files in .case format"""

    knowledge_base_path: str = "kg_graph.pkl"
    """Advanced version of graph-based RAG"""

    rag_path: str = "git_ignore_folder/kaggle_vector_base.pkl"
    """Base version of vector-based RAG"""

    ## proposal
    # (TODO: should goto sub config of proposal)
    #  Move to hypothesis_gen as a sub config instead of global config
    if_action_choosing_based_on_UCB: bool = False
    """Enable decision mechanism based on UCB algorithm"""

    if_using_vector_rag: bool = False
    """Enable basic vector-based RAG"""

    if_using_graph_rag: bool = False
    """Enable advanced graph-based RAG"""


KAGGLE_IMPLEMENT_SETTING = KaggleBasePropSetting()
