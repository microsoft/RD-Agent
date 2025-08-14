import json
from pathlib import Path

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
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
        # Get dataset and model info from global settings
        dataset = FT_RD_SETTING.dataset
        model_name = FT_RD_SETTING.base_model_name

        if not dataset:
            raise ValueError("Dataset must be specified in args of loop.py")
        if not model_name:
            raise ValueError("Model name must be specified in args of loop.py")

        # Basic attributes (align with downstream expectations)
        from rdagent.scenarios.finetune.utils import prev_model_dirname

        self.task = prev_model_dirname(model_name, dataset)
        self.competition = dataset  # Required by parent class and other methods

        # Set working directory (for backward compatibility if needed)
        # In Docker environments, this is handled by volume mounting
        self.debug_path = str(Path(FT_RD_SETTING.local_data_path) / dataset)

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
    def dataset(self) -> str:
        # Align naming for LLM scenario; the base class uses `competition`.
        return self.competition

    def _get_data_folder_description(self) -> str:
        """Generate folder structure description for fine-tuning."""
        return build_folder_description(self.competition)

    def _get_description(self) -> str:
        """Generate comprehensive task description combining dataset and model information."""
        dataset_info = extract_dataset_info(self.competition)
        model_info = extract_model_info()
        return build_finetune_description(dataset_info, model_info)

    def _analysis_dataset_description(self):
        """Analyze dataset and model for fine-tuning task characteristics."""
        dataset_info = extract_dataset_info(self.competition)
        model_info = extract_model_info()

        # Build analysis prompt
        analysis_content = f"Fine-tuning {model_info['name']} on {dataset_info['name']}"
        if dataset_info.get("samples"):
            analysis_content += (
                f"\n\nSample data:\n{json.dumps(dataset_info['samples'][0], ensure_ascii=False, indent=2)}"
            )

        try:
            sys_prompt = T(".prompts:dataset_description_template.system").r()
            user_prompt = T(".prompts:dataset_description_template.user").r(
                raw_description=analysis_content,
                data_folder_description=self.processed_data_folder_description,
            )

            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                json_mode=True,
                json_target_type=dict,
            )
            parsed = json.loads(response)
        except Exception as e:
            logger.warning(f"Analysis failed: {e}")
            parsed = {}

        # Set attributes with fine-tuning defaults
        self.task_type = parsed.get("Task Type", "LLM Fine-tuning")
        self.data_type = parsed.get("Data Type", "Text (Natural Language)")
        self.brief_description = parsed.get(
            "Brief Description",
            f"Fine-tune {model_info['name']} on {dataset_info['name']}",
        )
        self.dataset_description = parsed.get("Dataset Description", dataset_info.get("description", ""))
        self.model_output_channel = parsed.get("Channels per Sample", 1)
        self.metric_description = parsed.get("Evaluation Metric Description", "Training and validation loss")
        self.metric_name = parsed.get("Metric Name", "loss")
        self.metric_direction_guess = parsed.get("Metric Direction", False)

        # Fine-tuning specific attributes
        self.base_model_name = model_info["name"]
        self.dataset_name = dataset_info["name"]

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
            base_model_name=getattr(self, "base_model_name", "Unknown"),
            model_output_channel=getattr(self, "model_output_channel", 1),
            metric_description=getattr(self, "metric_description", "Training loss"),
        )

    def get_competition_full_desc(self) -> str:
        return T(".prompts:scenario_description").r(
            raw_description=self.raw_description,
        )

    def get_scenario_all_desc(self, eda_output=None) -> str:
        return T(".prompts:scenario_description").r(
            raw_description=self.raw_description,
            data_folder_description=self.processed_data_folder_description,
            metric_name=getattr(self, "metric_name", None),
            metric_direction=getattr(self, "metric_direction_guess", True),
            time_limit=getattr(self, "real_full_timeout", lambda: None)(),
        )

    def _initialize_scenario_data(self) -> None:
        """Initialize scenario descriptions and analysis."""
        self.raw_description = self._get_description()
        self.processed_data_folder_description = self._get_data_folder_description()
        self._analysis_dataset_description()
