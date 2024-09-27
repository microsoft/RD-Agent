from pathlib import Path

from pydantic_settings import BaseSettings

from rdagent.components.workflow.conf import BasePropSetting


class KaggleBasePropSetting(BasePropSetting):
    class Config:
        env_prefix = "KG_"
        """Use `KG_` as prefix for environment variables"""
        protected_namespaces = ()
        """Do not allow overriding of these namespaces"""

    # 1) overriding the default
    scen: str = "rdagent.scenarios.kaggle.experiment.scenario.KGScenario"
    """Scenario class for data mining model"""

    knowledge_base: str = ""  # TODO enable this line to use the knowledge base
    # knowledge_base: str = "rdagent.scenarios.kaggle.knowledge_management.graph.KGKnowledgeGraph"
    """Knowledge base class"""

    knowledge_base_path: str = "kg_graph.pkl"
    """Knowledge base path"""

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

    summarizer: str = "rdagent.scenarios.kaggle.developer.feedback.KGHypothesisExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""

    competition: str = ""

    local_data_path: str = "/data/userdata/share/kaggle"

    domain_knowledge_path: str = "/data/userdata/share/kaggle/domain_knowledge"

    rag_path: str = "git_ignore_folder/rag"

    if_action_choosing_based_on_UCB: bool = False

    if_using_graph_rag: bool = False

    if_using_vector_rag: bool = False

    auto_submit: bool = True


KAGGLE_IMPLEMENT_SETTING = KaggleBasePropSetting()
