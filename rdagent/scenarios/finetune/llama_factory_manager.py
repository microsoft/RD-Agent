"""LLaMA Factory Information Manager"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.core.experiment import FBWorkspace

logger = logging.getLogger(__name__)


class LLaMAFactoryManager:
    """LLaMA Factory Information Manager - Unified retrieval of LLaMA Factory related information

    Extracts fresh information once per instance to ensure data consistency within a session.
    Each new instance gets the latest LLaMA Factory information.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize manager

        Args:
            cache_dir: Cache directory, defaults to .llama_factory_info under current directory
        """
        self.cache_dir = cache_dir or Path(".llama_factory_info")
        self.cache_dir.mkdir(exist_ok=True)

        # Define directory structure
        self.basic_dir = self.cache_dir / "basic"
        self.params_dir = self.cache_dir / "parameters"
        self.metadata_file = self.cache_dir / "metadata.json"

        # Simple instance-level cache to avoid repeated extraction in same session
        self._info_cache: Optional[Dict] = None

    def extract_info_from_docker(self) -> Dict:
        """
        Extract LLaMA Factory information from Docker environment

        Always fetches fresh information to ensure up-to-date data.

        Returns:
            Complete LLaMA Factory information dictionary
        """
        logger.info("Extracting LLaMA Factory information from Docker environment...")

        try:
            workspace = FBWorkspace()

            # Use unified extraction script
            script_path = (
                Path(__file__).parent.parent.parent
                / "components"
                / "coder"
                / "finetune"
                / "extract_llama_factory_complete.txt"
            )
            workspace.inject_files(**{"extract_complete.py": script_path.read_text()})

            # Build llama factory info volume configuration
            base_path = Path(FT_RD_SETTING.file_path)
            llama_info_dir = base_path / ".llama_factory_info"
            llama_info_dir.mkdir(exist_ok=True)
            llama_factory_volumes = {str(llama_info_dir): {"bind": "/workspace/.llama_factory_info", "mode": "rw"}}

            result = workspace.run(
                env=get_ft_env(extra_volumes=llama_factory_volumes, running_timeout_period=120),
                entry="python extract_complete.py",
            )

            if result.exit_code == 0:
                # Files are directly written to mounted directory, load them immediately
                info_data = self._load_from_structured_files()
                if info_data:
                    self._info_cache = info_data  # Cache for this instance
                    logger.info("Successfully extracted LLaMA Factory information")
                    return info_data
                else:
                    raise RuntimeError("Failed to load information from structured files")
            else:
                raise RuntimeError(f"Information extraction failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Failed to extract LLaMA Factory information from Docker: {e}")
            # Try to use existing data as fallback
            if self.metadata_file.exists():
                logger.warning("Attempting to use existing data as fallback")
                fallback_data = self._load_from_structured_files()
                self._info_cache = fallback_data  # Cache fallback data too
                return fallback_data
            raise RuntimeError(f"Unable to extract LLaMA Factory information: {e}")

    def _load_from_structured_files(self) -> Dict:
        """Load information from structured files"""
        try:
            info_data = {}

            # Load basic information
            if self.basic_dir.exists():
                basic_files = ["methods.json", "models.json", "peft_methods.json", "training_stages.json"]
                for file_name in basic_files:
                    file_path = self.basic_dir / file_name
                    if file_path.exists():
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            key = "supported_models" if file_name == "models.json" else file_name.replace(".json", "")
                            info_data[key] = data

            # Load parameter information
            if self.params_dir.exists():
                parameters = {}
                param_types = ["data", "model", "training", "finetuning"]
                for param_type in param_types:
                    param_file = self.params_dir / f"{param_type}.json"
                    if param_file.exists():
                        with open(param_file, "r", encoding="utf-8") as f:
                            parameters[param_type] = json.load(f)
                if parameters:
                    info_data["parameters"] = parameters

            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    info_data.update(metadata)

            return info_data

        except Exception as e:
            logger.error(f"Failed to load structured files: {e}")
            logger.debug(f"Cache directory: {self.cache_dir}, exists: {self.cache_dir.exists()}")
            return {}

    def get_info(self, force_refresh: bool = False) -> Dict:
        """Get complete LLaMA Factory information - extracts once per instance"""
        if self._info_cache is None or force_refresh:
            self._info_cache = self.extract_info_from_docker()
        return self._info_cache

    # === Query Methods ===

    @property
    def methods(self) -> List[str]:
        """List of available fine-tuning methods"""
        return self.get_info().get("methods", [])

    @property
    def models(self) -> List[str]:
        """List of available base models"""
        return list(self.get_info().get("supported_models", {}).keys())

    @property
    def peft_methods(self) -> List[str]:
        """List of PEFT methods"""
        return self.get_info().get("peft_methods", [])

    @property
    def training_stages(self) -> Dict[str, str]:
        """Training stage mapping"""
        return self.get_info().get("training_stages", {})

    def is_peft_method(self, method: str) -> bool:
        """Check if method is a PEFT method"""
        return method in self.peft_methods

    def get_parameters(self, param_type: str = None) -> Dict:
        """
        Get parameter information

        Args:
            param_type: Parameter type ("data", "model", "training", "finetuning"), None means all
        """
        all_params = self.get_info().get("parameters", {})
        return all_params.get(param_type, {}) if param_type else all_params

    def format_method_params(self, method: str) -> str:
        """Format parameter information for specified method as prompt format"""
        lines = [f"## {method.upper()} Fine-tuning Method Parameters"]

        # Get core parameter types
        core_params = {
            "data": ["dataset", "cutoff_len", "preprocessing_num_workers"],
            "model": ["model_name_or_path", "torch_dtype", "device_map"],
            "training": [
                "learning_rate",
                "num_train_epochs",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "save_steps",
                "logging_steps",
            ],
            "finetuning": ["stage", "finetuning_type", "cutoff_len"],
        }

        # Add method-specific parameters
        if method.lower() == "lora":
            core_params["finetuning"].extend(["lora_rank", "lora_alpha", "lora_dropout", "lora_target"])
        elif method.lower() == "qlora":
            core_params["finetuning"].extend(["lora_rank", "lora_alpha", "lora_dropout", "lora_target"])
            core_params["model"].extend(["quantization_bit", "quantization_type"])

        all_params = self.get_parameters()

        for param_type, param_names in core_params.items():
            type_params = all_params.get(param_type, {})
            if not type_params:
                continue

            lines.append(f"\n### {param_type.upper()} Parameters:")
            for param_name in param_names:
                if param_name in type_params:
                    param_info = type_params[param_name]
                    simple_format = param_info.get("simple_format", f'"{param_name}"')
                    lines.append(f"- {simple_format}")

        return "\n".join(lines)

    def get_metadata_info(self) -> Dict:
        """Get metadata information for debugging"""
        if not self.metadata_file.exists():
            return {"has_metadata": False}

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                timestamp = metadata.get("timestamp", 0)
                commit_sha = metadata.get("commit_sha")

                return {
                    "has_metadata": True,
                    "timestamp": timestamp,
                    "age_hours": (time.time() - timestamp) / 3600,
                    "last_updated": metadata.get("last_updated", "Unknown"),
                    "commit_sha": commit_sha[:8] if commit_sha else None,
                }
        except Exception as e:
            return {"has_metadata": False, "error": f"Unable to read metadata file: {e}"}
