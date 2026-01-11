import json
import os
from pathlib import Path
import shutil

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.finetune.benchmark.benchmark import run_benchmark
from rdagent.scenarios.finetune.datasets import prepare_all
from rdagent.scenarios.finetune.experiment.workspace import FTWorkspace
from rdagent.scenarios.finetune.scen.llama_factory_manager import LLaMAFactory_manager
from rdagent.scenarios.finetune.scen.memory_estimator import MemoryEstimator
from rdagent.scenarios.finetune.scen.utils import (
    FinetuneDatasetDescriptor,
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

        # NOTE: we disable the cache for environment. in case of changing cuda config
        self.device_info = get_runtime_environment_by_env(get_ft_env(enable_cache=False))
        self.gpu_count = self._get_gpu_count()
        self.model_info = FinetuneDatasetDescriptor().describe_model(self.base_model)

        # Initialize memory estimator
        self.memory_report = self._generate_memory_report()

        self.baseline_benchmark_score = self.run_baseline_model_evaluation(model_name=self.base_model, benchmark_name=self.target_benchmark)

    
    def _get_gpu_count(self) -> int:
        """Return GPU count parsed from device_info stored at initialization."""
        gpu_info = json.loads(self.device_info).get("gpu", {})
        if gpu_info.get("source") == "pytorch":
            return gpu_info.get("gpu_count", 0)
        elif "gpus" in gpu_info:
            return len(gpu_info["gpus"])

    def benchmark_hash(self, model_name, benchmark_name) -> str:
        return f"llm_finetune_baseline_eval_{model_name}_{benchmark_name}"

    @cache_with_pickle(benchmark_hash)
    def run_baseline_model_evaluation(self, model_name, benchmark_name) -> dict:
        ws = FTWorkspace()
        shutil.copytree(
            Path(FT_RD_SETTING.file_path) / "models" / model_name,
            ws.workspace_path / "models" / model_name,
            dirs_exist_ok=True,
        )
        bm = run_benchmark(
            workspace_path=str(ws.workspace_path),
            model_path=ws.workspace_path / "models" / model_name,
            model_name=model_name,
            benchmark_name=benchmark_name,
            gpu_count=self.gpu_count,
        )
        return bm

    def real_full_timeout(self):
        return FT_RD_SETTING.full_timeout

    def _generate_memory_report(self) -> str:
        """Generate memory estimation report based on hardware and model."""
        try:
            # Parse device info
            device_info = json.loads(self.device_info) if isinstance(self.device_info, str) else self.device_info
            gpu_info = device_info.get("gpu", {})

            # Extract GPU info based on source
            if gpu_info.get("source") == "pytorch":
                # PyTorch format: has gpu_count and total_gpu_memory_gb directly
                num_gpus = gpu_info.get("gpu_count")
                gpu_mem = gpu_info.get("total_gpu_memory_gb")
            else:
                # nvidia-smi format: has gpus array with memory_total_mb
                gpus = gpu_info.get("gpus", [])
                num_gpus = len(gpus) if gpus else None
                gpu_mem = gpus[0].get("memory_total_mb", 0) / 1024 if gpus else None  # MB -> GB

            # Skip if GPU info not available
            if not num_gpus or not gpu_mem:
                logger.warning("GPU info not available, skipping memory report")
                return ""

            # Create estimator from model name (pass model_specs for max_position_embeddings)
            estimator = MemoryEstimator.from_model_name(
                name=self.base_model,
                gpu_mem=gpu_mem,
                num_gpus=num_gpus,
                model_specs=self.model_info.get("specs", ""),
            )
            return estimator.format()
        except Exception as e:
            logger.warning(f"Failed to generate memory report: {e}")
            return ""

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
        # Pass dataset_config directly - it already has the unified tasks structure
        dataset_summaries = [
            {
                "name": ds_name,
                "total_samples": ds_config.get("total_samples"),
                "total_size_mb": ds_config.get("total_size_mb"),
                "tasks": ds_config.get("tasks", {}),
                "readme": ds_config.get("readme"),
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

                # Only keep entries that have corresponding local directories
                local_datasets = {d.name for d in datasets_dir.iterdir() if d.is_dir() and not d.name.startswith(".")}
                existing_config = {k: v for k, v in existing_config.items() if k in local_datasets}

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

    def get_scenario_all_desc(self, enable_dataset_description: bool = False) -> str:
        """Get complete scenario description for LLM fine-tuning.

        Uses dataset_config as the single source of truth for dataset information.
        The prompt template renders tasks with their statistics and samples.
        """
        return T(".prompts:scenario_description").r(
            user_target_scenario=self.user_target_scenario,
            target_benchmark=self.target_benchmark,
            benchmark_description=self.benchmark_description,
            device_info=self.device_info,
            memory_report=self.memory_report,
            chosen_model=FT_RD_SETTING.base_model is not None,
            base_model=FT_RD_SETTING.base_model,
            dataset_config=self.dataset_config,
            model_info=self.model_info,
            full_timeout=f"{self.real_full_timeout() / 60 / 60:.2f} hours",
            enable_dataset_description=enable_dataset_description,
            upper_data_size_limit=FT_RD_SETTING.upper_data_size_limit,
        )
