import json
import os
from pathlib import Path

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.finetune.datasets import prepare_all
from rdagent.scenarios.finetune.scen.llama_factory_manager import LLaMAFactory_manager
from rdagent.scenarios.finetune.scen.utils import (
    FinetuneDatasetDescriptor,
    _truncate_long_values,
    generate_dataset_info_config,
)
from rdagent.scenarios.finetune.utils import ensure_ft_assets_exist
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T


class LLMFinetuneScen(DataScienceScen):
    """LLMFinetuneScen Scenario"""

    def __init__(self) -> None:
        """Initialize LLM finetune scenario using configuration from FT_RD_SETTING."""
        logger.info("Initializing LLM Fine-tune scenario")

        # Basic attributes
        self.user_target_scenario = FT_RD_SETTING.user_target_scenario
        self.target_benchmark = FT_RD_SETTING.target_benchmark
        self.benchmark_description = FT_RD_SETTING.benchmark_description
        self.dataset = FT_RD_SETTING.dataset
        self.base_model = FT_RD_SETTING.base_model

        # Validate and prepare environment
        self._validate_and_prepare_environment()

        # Initialize LLaMA Factory manager
        self._initialize_llama_factory()

        # Generate dataset configuration for all datasets first
        self.dataset_config = self._prepare_dataset_config()

        # Select relevant datasets based on user target scenario (using full config info)
        self.selected_datasets = self._select_relevant_datasets()

        # Filter dataset_config to only include selected datasets
        self.dataset_config = {k: v for k, v in self.dataset_config.items() if k in self.selected_datasets}

        # timeout tracking
        self.timeout_increase_count = 0

        self.device_info = get_runtime_environment_by_env(get_ft_env())
        self.model_info = FinetuneDatasetDescriptor().describe_model(self.base_model)

    def real_debug_timeout(self):
        return FT_RD_SETTING.debug_timeout

    def real_full_timeout(self):
        return FT_RD_SETTING.full_timeout

    def _validate_and_prepare_environment(self):
        """Validate FT_FILE_PATH and prepare all registered datasets"""
        ft_root = Path(FT_RD_SETTING.file_path)
        if not ft_root.exists():
            os.makedirs(ft_root, mode=0o777, exist_ok=True)
            logger.info(f"FT_FILE_PATH not exists, created FT_FILE_PATH directory: {ft_root}")

        # Prepare all registered datasets
        prepare_all()

        # Ensure model assets exist
        if self.base_model:
            ensure_ft_assets_exist(model=self.base_model, check_model=True)

    def _initialize_llama_factory(self):
        """Initialize LLaMA Factory information manager"""

        # Extract LLaMA Factory information (pulls latest code automatically)
        info = LLaMAFactory_manager.get_info()

        # Log extracted information
        methods_count = len(info.get("methods", []))
        params_count = sum(len(p) if isinstance(p, dict) else 0 for p in info.get("parameters", {}).values())
        logger.info(f"LLaMA Factory initialized: {methods_count} methods, {params_count} parameters")

    def _select_relevant_datasets(self) -> list[str]:
        """Select relevant datasets based on user target scenario using LLM.

        Uses self.dataset_config which contains full information (stats, description, samples).
        """
        total = len(self.dataset_config)

        # If user specified a dataset, use it directly
        if self.dataset:
            selected, reasoning = [self.dataset], "User specified dataset directly"
        elif not self.dataset_config:
            logger.warning("No datasets found for selection")
            return []
        else:
            # Use LLM to select relevant datasets
            logger.info(f"Found {total} datasets, selecting relevant ones...")
            selected, reasoning = self._llm_select_datasets()

        # Log results
        logger.info(f"Dataset selection: {len(selected)}/{total} - {selected}")
        logger.log_object(
            {"selected_datasets": selected, "total_datasets": total, "reasoning": reasoning},
            tag="dataset_selection",
        )
        return selected

    def _llm_select_datasets(self) -> tuple[list[str], str]:
        """Use LLM to select relevant datasets."""
        dataset_summaries = [
            {
                "name": ds_name,
                "stats": ds_config.get("stats"),
                "readme": ds_config.get("readme"),
                "description": ds_config.get("description"),
                "first_sample": (
                    _truncate_long_values(ds_config["samples"][0], max_length=500) if ds_config.get("samples") else None
                ),
            }
            for ds_name, ds_config in self.dataset_config.items()
        ]

        system_prompt = T(".prompts:dataset_selection.system").r(
            user_target_scenario=self.user_target_scenario,
            target_benchmark=self.target_benchmark,
            benchmark_description=self.benchmark_description,
        )
        user_prompt = T(".prompts:dataset_selection.user").r(datasets=dataset_summaries)

        response = APIBackend().build_messages_and_create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
        )

        result = json.loads(response)
        return result.get("selected_datasets", []), result.get("reasoning", "")

    def _prepare_dataset_config(self) -> dict:
        """Generate dataset_info.json configuration.

        This is the single source of truth for dataset information, containing:
        - LlamaFactory compatible fields (file_name, formatting, columns)
        - Auto-computed statistics (stats.column_stats)
        - Data samples (truncated)
        - AI-generated description

        Returns:
            dict: Complete dataset configuration
        """
        datasets_dir = Path(FT_RD_SETTING.file_path) / "datasets"
        dataset_info_path = datasets_dir / "dataset_info.json"

        # Check if already configured
        existing_config = {}
        if dataset_info_path.exists():
            try:
                with open(dataset_info_path, "r", encoding="utf-8") as f:
                    existing_config = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing dataset_info.json: {e}")

        # Generate config for all datasets (will be filtered later by _select_relevant_datasets)
        target_dataset_list = [] if self.dataset is None else [self.dataset]
        logger.info(
            f"Generating dataset_info.json configuration for: {target_dataset_list if target_dataset_list else 'all datasets'}"
        )
        generated_config = generate_dataset_info_config(target_dataset_list, FT_RD_SETTING.file_path, existing_config)
        for dataset_name, config in generated_config.items():
            existing_config[dataset_name] = config

        try:
            os.makedirs(datasets_dir, mode=0o777, exist_ok=True)

            with open(dataset_info_path, "w", encoding="utf-8") as f:
                json.dump(existing_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully updated dataset_info.json with configuration for: {target_dataset_list}")
        except Exception as e:
            raise RuntimeError(f"Failed to write dataset_info.json: {e}")
        return existing_config

    @property
    def metric_direction(self) -> bool:
        """Metric direction for LLM fine-tuning (higher is better)"""
        return True

    def get_scenario_all_desc(self, enable_dataset_description: bool = True) -> str:
        """Get complete scenario description for LLM fine-tuning.

        Uses dataset_config as the single source of truth for dataset information.
        The prompt template selectively renders only needed fields (excluding formatting, columns).
        """
        # Add first_sample (truncated to 500 chars) for each dataset
        prompt_config = {
            ds_name: {
                **ds_config,
                "first_sample": (
                    _truncate_long_values(ds_config["samples"][0], max_length=500) if ds_config.get("samples") else None
                ),
            }
            for ds_name, ds_config in self.dataset_config.items()
        }

        return T(".prompts:scenario_description").r(
            user_target_scenario=self.user_target_scenario,
            target_benchmark=self.target_benchmark,
            benchmark_description=self.benchmark_description,
            device_info=self.device_info,
            chosen_model=FT_RD_SETTING.base_model is not None,
            base_model=FT_RD_SETTING.base_model,
            dataset_config=prompt_config,
            model_info=self.model_info,
            debug_timeout=f"{self.real_debug_timeout() / 60:.2f} minutes",
            full_timeout=f"{self.real_full_timeout() / 60 / 60:.2f} hours",
            enable_dataset_description=enable_dataset_description,
        )
