import os

from pydantic_settings import SettingsConfigDict

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.conf import RD_AGENT_SETTINGS, ExtendedBaseSettings


class DSFinetuneScen(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="FT_", protected_namespaces=())
    scen: str = "rdagent.app.finetune.data_science.scen.DSFinetuneScen"
    """
    Scenario class for data science tasks.
    - For Kaggle competitions, use: "rdagent.scenarios.data_science.scen.KaggleScen"
    - For custom data science scenarios, use: "rdagent.scenarios.data_science.scen.DataScienceScen"
    - For LLM finetune scenarios, use: "rdagent.app.finetune.llm.scen.LLMFinetuneScen"
    - For Data science finetune scenarios, use: "rdagent.app.finetune.data_science.scen.DSFinetuneScen"
    """

    debug_timeout: int = 3600
    """The timeout limit for running on debugging data"""
    full_timeout: int = 10800
    """The timeout limit for running on full data"""

    coder_on_whole_pipeline: bool = True
    enable_model_dump: bool = True
    app_tpl: str = "app/finetune/data_science/tpl"


def update_settings(competition: str):
    """
    Update the RD_AGENT_SETTINGS with the values from DS_FINETUNE_SETTINGS.
    """
    DS_FINETUNE_SETTINGS = DSFinetuneScen()
    RD_AGENT_SETTINGS.app_tpl = DS_FINETUNE_SETTINGS.app_tpl
    os.environ["DS_CODER_COSTEER_EXTRA_EVALUATOR"] = '["rdagent.app.finetune.share.eval.PrevModelLoadEvaluator"]'
    for field_name, new_value in DS_FINETUNE_SETTINGS.model_dump().items():
        if hasattr(DS_RD_SETTING, field_name):
            setattr(DS_RD_SETTING, field_name, new_value)
    DS_RD_SETTING.competition = competition
