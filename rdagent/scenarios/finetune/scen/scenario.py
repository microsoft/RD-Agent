from pathlib import Path

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.finetune.scen.utils import (
    build_finetune_description,
    build_folder_description,
    extract_dataset_info,
    extract_model_info,
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

        # Set working directory (for backward compatibility if needed)
        # In Docker environments, this is handled by volume mounting
        self.debug_path = str(Path(FT_RD_SETTING.local_data_path) / self.dataset)

        # Initialize descriptions and analysis
        self._initialize_scenario_data()

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

    @property
    def rich_style_description(self) -> str:
        raise NotImplementedError

    @property
    def background(self) -> str:
        return T(".prompts:finetune_background").r(
            task_type=getattr(self, "task_type", "LLM Fine-tuning"),
            data_type=getattr(self, "data_type", "Text (Natural Language)"),
            brief_description=getattr(self, "brief_description", "Fine-tuning task"),
            dataset_description=getattr(self, "dataset_description", ""),
            base_model_name=getattr(self, "base_model", "Unknown"),
        )

    def get_scenario_all_desc(self, eda_output=None) -> str:
        return T(".prompts:scenario_description").r(
            raw_description=self.raw_description,
            data_folder_description=self.processed_data_folder_description,
            time_limit=getattr(self, "real_full_timeout", lambda: None)(),
        )

    def _initialize_scenario_data(self) -> None:
        """Initialize scenario descriptions - simplified for LLaMA Factory."""
        # Get basic descriptions (reuse extracted info to avoid redundancy)
        dataset_info = extract_dataset_info(self.dataset)
        model_info = extract_model_info(self.base_model)

        self.raw_description = build_finetune_description(dataset_info, model_info)
        # TODO: data process is not implemented yet
        self.processed_data_folder_description = build_folder_description(self.dataset)

        # Set minimal required attributes without expensive API calls
        self.task_type = "LLM Fine-tuning"
        self.data_type = "Text (Natural Language)"
        self.brief_description = f"Fine-tune {model_info['name']} on {dataset_info['name']} using LLaMA Factory"
        self.dataset_description = dataset_info.get("description", "")
