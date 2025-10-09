import json
import os
from pathlib import Path

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.finetune.scen.utils import (
    build_finetune_description,
    build_folder_description,
    extract_dataset_info,
    extract_model_info,
    generate_dataset_info_config,
)
from rdagent.scenarios.finetune.utils import ensure_ft_assets_exist
from rdagent.utils.agent.tpl import T

from .llama_factory_manager import get_llama_factory_manager


class LLMFinetuneScen(DataScienceScen):
    """LLMFinetuneScen Scenario"""

    def __init__(self) -> None:
        """Initialize LLM finetune scenario using configuration from FT_RD_SETTING."""
        logger.info("Initializing LLM Fine-tune scenario")

        # Basic attributes
        self.dataset = FT_RD_SETTING.dataset
        self.base_model = FT_RD_SETTING.base_model
        self.task = FT_RD_SETTING.task

        # Validate and prepare environment
        self._validate_and_prepare_environment()

        # Initialize LLaMA Factory manager
        self._initialize_llama_factory()

        # Generate dataset configuration
        self._prepare_dataset_info()

        # timeout tracking
        self.timeout_increase_count = 0

        logger.info(f"LLM Fine-tune scenario initialized for dataset='{self.dataset}', model='{self.base_model}'")

    def real_debug_timeout(self):
        return FT_RD_SETTING.debug_timeout

    def recommend_debug_timeout(self):
        return FT_RD_SETTING.debug_recommend_timeout

    def real_full_timeout(self):
        return FT_RD_SETTING.full_timeout

    def recommend_full_timeout(self):
        return FT_RD_SETTING.full_recommend_timeout

    def _validate_and_prepare_environment(self):
        """Validate FT_FILE_PATH and ensure required assets exist"""
        # Validate FT_FILE_PATH environment variable
        if not FT_RD_SETTING.file_path:
            raise Exception("Please set FT_FILE_PATH environment variable")

        ft_root = Path(FT_RD_SETTING.file_path)
        if not ft_root.exists():
            os.makedirs(ft_root, mode=0o777, exist_ok=True)
            logger.info(f"FT_FILE_PATH not exists, created FT_FILE_PATH directory: {ft_root}")

        # Ensure dataset assets exist
        ensure_ft_assets_exist(dataset=self.dataset, check_dataset=True)

        # Ensure model assets exist if model is specified
        if self.base_model:
            ensure_ft_assets_exist(model=self.base_model, check_model=True)

    def _initialize_llama_factory(self):
        """Initialize LLaMA Factory information manager"""
        self.llama_factory_manager = get_llama_factory_manager()
        
        # Extract LLaMA Factory information (pulls latest code automatically)
        info = self.llama_factory_manager.get_info()
        
        # Log extracted information
        methods_count = len(info.get("methods", []))
        params_count = sum(len(p) if isinstance(p, dict) else 0 for p in info.get("parameters", {}).values())
        logger.info(f"LLaMA Factory initialized: {methods_count} methods, {params_count} parameters")

    def _prepare_dataset_info(self):
        """Generate dataset_info.json configuration"""
        if not self.dataset:
            return

        datasets_dir = Path(FT_RD_SETTING.file_path) / "datasets"
        dataset_info_path = datasets_dir / "dataset_info.json"

        # Check if already configured
        existing_config = {}
        if dataset_info_path.exists():
            try:
                with open(dataset_info_path, "r", encoding="utf-8") as f:
                    existing_config = json.load(f)
                if self.dataset in existing_config:
                    logger.info(f"Dataset '{self.dataset}' already configured in dataset_info.json, skipping")
                    return
            except Exception as e:
                logger.warning(f"Failed to load existing dataset_info.json: {e}")

        # Generate new configuration
        logger.info(f"Generating dataset_info.json configuration for dataset '{self.dataset}'")
        generated_config = generate_dataset_info_config(self.dataset, FT_RD_SETTING.file_path)
        existing_config[self.dataset] = generated_config

        try:
            os.makedirs(datasets_dir, mode=0o777, exist_ok=True)

            with open(dataset_info_path, "w", encoding="utf-8") as f:
                json.dump(existing_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully updated dataset_info.json with configuration for '{self.dataset}'")
        except Exception as e:
            raise RuntimeError(f"Failed to write dataset_info.json: {e}")

    def _get_data_folder_description(self) -> str:
        """Generate folder description by running describe_data_folder_v2 inside Docker environment"""
        return build_folder_description(self.dataset)
