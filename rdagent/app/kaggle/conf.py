from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class KaggleBasePropSetting(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="KG_", protected_namespaces=())

    # 1) overriding the default
    scen: str = "rdagent.scenarios.kaggle.experiment.scenario.KGScenario"
    """Scenario class for data mining model"""

    hypothesis_gen: str = "rdagent.scenarios.kaggle.proposal.proposal.KGHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.kaggle.proposal.proposal.KGHypothesis2Experiment"
    """Hypothesis to experiment class"""

    feature_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGFactorCoSTEER"
    """Feature Coder class"""

    model_feature_selection_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGModelFeatureSelectionCoder"
    """Model Feature Selection Coder class"""

    model_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGModelCoSTEER"
    """Model Coder class"""

    feature_runner: str = "rdagent.scenarios.kaggle.developer.runner.KGFactorRunner"
    """Feature Runner class"""

    model_runner: str = "rdagent.scenarios.kaggle.developer.runner.KGModelRunner"
    """Model Runner class"""

    summarizer: str = "rdagent.scenarios.kaggle.developer.feedback.KGExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""

    competition: str = ""
    """Kaggle competition name, e.g., 'sf-crime'"""

    template_path: str = "rdagent/scenarios/kaggle/experiment/templates"
    """Kaggle competition base templates path"""

    local_data_path: str = ""
    """Folder storing Kaggle competition data"""

    # Evaluation on Test related
    if_using_mle_data: bool = False
    auto_submit: bool = False
    """Automatically upload and submit each experiment result to Kaggle platform"""

    # Conditionally set the knowledge_base based on the use of graph RAG
    knowledge_base: str = ""
    """Knowledge base class, uses 'KGKnowledgeGraph' when advanced graph-based RAG is enabled, otherwise empty."""
    if_action_choosing_based_on_UCB: bool = False
    """Enable decision mechanism based on UCB algorithm"""

    domain_knowledge_path: str = "/data/userdata/share/kaggle/domain_knowledge"
    """Folder storing domain knowledge files in .case format"""

    knowledge_base_path: str = "kg_graph.pkl"
    """Advanced version of graph-based RAG"""

    rag_path: str = "git_ignore_folder/kaggle_vector_base.pkl"
    """Base version of vector-based RAG"""

    if_using_vector_rag: bool = False
    """Enable basic vector-based RAG"""

    if_using_graph_rag: bool = False
    """Enable advanced graph-based RAG"""

    mini_case: bool = False
    """Enable mini-case study for experiments"""


KAGGLE_IMPLEMENT_SETTING = KaggleBasePropSetting()
