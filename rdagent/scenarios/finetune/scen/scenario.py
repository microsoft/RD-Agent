from pathlib import Path

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.finetune.scen.utils import (
    build_finetune_description,
    build_folder_description,
    extract_dataset_info,
    extract_model_info,
    get_unified_mount_volumes,
)
from rdagent.utils.agent.tpl import T


class LLMFinetuneScen(DataScienceScen):
    """LLMFinetuneScen Scenario"""

    def __init__(self) -> None:
        """Initialize LLM finetune scenario using configuration from FT_RD_SETTING."""

        # Basic attributes (align with downstream expectations)
        from rdagent.scenarios.finetune.utils import prev_model_dirname

        self.dataset = FT_RD_SETTING.dataset
        self.base_model = FT_RD_SETTING.base_model_name
        self.task = prev_model_dirname(self.base_model, self.dataset)

        # timeout tracking
        self.timeout_increase_count = 0

    def real_debug_timeout(self):
        return FT_RD_SETTING.debug_timeout

    def recommend_debug_timeout(self):
        return FT_RD_SETTING.debug_recommend_timeout

    def real_full_timeout(self):
        return FT_RD_SETTING.full_timeout

    def recommend_full_timeout(self):
        return FT_RD_SETTING.full_recommend_timeout

    def _get_data_folder_description(self) -> str:
        """Generate folder description by running describe_data_folder_v2 inside Docker environment."""
        return build_folder_description(self.dataset)
