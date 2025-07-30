from pydantic_settings import SettingsConfigDict

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.conf import RD_AGENT_SETTINGS, ExtendedBaseSettings


class LLMFinetuneScen(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="FT_", protected_namespaces=())
    scen: str = "rdagent.app.finetune.scen.LLMFinetuneScen"
    """
    Scenario class for data science tasks.
    - For Kaggle competitions, use: "rdagent.scenarios.data_science.scen.KaggleScen"
    - For custom data science scenarios, use: "rdagent.scenarios.data_science.scen.DataScienceScen"
    - For LLM finetune scenarios, use: "rdagent.app.finetune.scen.LLMFinetuneScen"
    - For Data science finetune scenarios, use: "rdagent.app.tune.scen.DSFinetuneScen"
    """

    hypothesis_gen: str = "rdagent.app.finetune.proposal.FinetuneExpGen"
    """Hypothesis generation class"""

    debug_timeout: int = 36000
    """The timeout limit for running on debugging data"""
    full_timeout: int = 360000
    """The timeout limit for running on full data"""

    coder_on_whole_pipeline: bool = True
    enable_model_dump: bool = True
    app_tpl: str = "app/finetune/tpl"


def init_finetune_settings(competition: str) -> None:
    """Initialize finetune settings"""
    LLM_FINETUNE_SETTINGS = LLMFinetuneScen()
    RD_AGENT_SETTINGS.app_tpl = LLM_FINETUNE_SETTINGS.app_tpl
    global DS_RD_SETTING
    DS_RD_SETTING = DS_RD_SETTING.model_copy(update=LLM_FINETUNE_SETTINGS.model_dump())
    DS_RD_SETTING.competition = competition
    return DS_RD_SETTING
