"""
LLaMA Factory manager for parameter extraction and caching.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger


class LLaMAFactoryManager:
    """Manager for LLaMA Factory information extraction and caching."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the manager instance."""
        if cache_dir is None:
            base_path = FT_RD_SETTING.file_path or str(Path.home() / ".rdagent")
            cache_dir = Path(base_path) / ".llama_factory_info"
        self.cache_dir = cache_dir
        self._info_cache: Optional[Dict] = None

    def extract_info_from_docker(self) -> Dict:
        """Extract LLaMA Factory information from Docker environment."""
        logger.info("Extracting LLaMA Factory information from Docker environment")

        # Prepare extraction script
        workspace = FBWorkspace()
        script_path = (
            Path(__file__).parent.parent.parent
            / "components"
            / "coder"
            / "finetune"
            / "extract_llama_factory_simple.py"
        )
        workspace.inject_files(**{"extract_script.py": script_path.read_text()})

        # Setup cache directory and Docker volumes
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        volumes = {str(self.cache_dir): {"bind": "/workspace/.llama_factory_info", "mode": "rw"}}

        # Run extraction
        result = workspace.run(
            env=get_ft_env(extra_volumes=volumes, running_timeout_period=120),
            entry="python extract_script.py",
        )

        if result.exit_code != 0:
            raise RuntimeError(f"Information extraction failed: {result.stdout}")

        # Load the extracted data
        self._info_cache = self._load_categorized_structure()
        if not self._info_cache:
            raise RuntimeError("Failed to load LLaMA Factory information")

        logger.info("Successfully extracted LLaMA Factory information")
        return self._info_cache

    def _load_categorized_structure(self) -> Dict:
        """Load categorized information from structured files."""
        data = {}

        # Load metadata
        metadata_file = self.cache_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, encoding="utf-8") as f:
                metadata = json.load(f)
                data.update(metadata)

        # Load constants
        constants_dir = self.cache_dir / "constants"
        if constants_dir.exists():
            for json_file in constants_dir.glob("*.json"):
                key = json_file.stem
                with open(json_file, encoding="utf-8") as f:
                    data[key] = json.load(f)

        # Load parameters
        data["parameters"] = {}

        # Load common parameters
        common_dir = self.cache_dir / "common"
        if common_dir.exists():
            for json_file in common_dir.glob("*.json"):
                key = json_file.stem
                with open(json_file, encoding="utf-8") as f:
                    data["parameters"][key] = json.load(f)

        # Merge method-specific parameters into finetuning
        method_dir = self.cache_dir / "method_specific"
        if method_dir.exists():
            finetune_params = {}
            for json_file in method_dir.glob("*.json"):
                with open(json_file, encoding="utf-8") as f:
                    finetune_params.update(json.load(f))
            data["parameters"]["finetuning"] = finetune_params

        return data

    def get_info(self, force_refresh: bool = False) -> Dict:
        """Get complete LLaMA Factory information."""
        if self._info_cache is None or force_refresh:
            # Try loading from cache first
            cached_data = self._load_categorized_structure()
            if cached_data and cached_data.get("parameters"):
                logger.info("Loaded LLaMA Factory info from cache")
                self._info_cache = cached_data
                return self._info_cache

            # Extract fresh information if cache is empty
            self._info_cache = self.extract_info_from_docker()

        return self._info_cache

    @property
    def methods(self) -> List[str]:
        """Available fine-tuning methods."""
        return self.get_info().get("methods", [])

    @property
    def models(self) -> List[str]:
        """Available base models."""
        return list(self.get_info().get("supported_models", {}).keys())

    @property
    def peft_methods(self) -> List[str]:
        """Available PEFT methods."""
        # Return hardcoded PEFT methods since they are fixed
        return ["lora"]

    @property
    def training_stages(self) -> Dict[str, str]:
        """Training stage mapping."""
        return self.get_info().get("training_stages", {})

    def is_peft_method(self, method: str) -> bool:
        """Check if the given method is a PEFT method."""
        return method in self.peft_methods

    def get_parameters(self, param_type: str = None) -> Dict:
        """Get parameters by type or all parameters."""
        params = self.get_info().get("parameters", {})

        if param_type:
            return params.get(param_type, {})
        return params

    def format_method_params(self, method: str) -> str:
        """Format parameters for a specific method as a readable string."""
        lines = [f"Parameters for {method} fine-tuning:"]

        # Get all parameters
        all_params = self.get_parameters()

        # Common parameters
        for param_type in ["model", "data", "training"]:
            if param_type in all_params:
                lines.append(f"\n### {param_type.upper()} Parameters:")
                type_params = all_params[param_type]
                for param_name, param_info in type_params.items():
                    if isinstance(param_info, dict) and "help" in param_info:
                        lines.append(f"- {param_name}: {param_info['help']}")

        # Method-specific parameters from finetuning
        if "finetuning" in all_params:
            finetune_params = all_params["finetuning"]

            # Filter parameters based on method
            if method == "lora":
                relevant_params = [p for p in finetune_params.keys() if "lora" in p or "additional_target" in p]
            elif method == "freeze":
                relevant_params = [p for p in finetune_params.keys() if "freeze" in p]
            else:
                relevant_params = []

            if relevant_params:
                lines.append(f"\n### {method.upper()} Specific Parameters:")
                for param_name in relevant_params:
                    if param_name in finetune_params:
                        param_info = finetune_params[param_name]
                        if isinstance(param_info, dict) and "help" in param_info:
                            lines.append(f"- {param_name}: {param_info['help']}")

        return "\n".join(lines)


# Module-level singleton instance
_manager_instance: Optional[LLaMAFactoryManager] = None


def get_llama_factory_manager(cache_dir: Optional[Path] = None) -> LLaMAFactoryManager:
    """Get the singleton LLaMAFactoryManager instance."""
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = LLaMAFactoryManager(cache_dir)
    elif cache_dir is not None and _manager_instance.cache_dir != cache_dir:
        logger.info(f"Updating cache directory from {_manager_instance.cache_dir} to {cache_dir}")
        _manager_instance.cache_dir = cache_dir
        _manager_instance._info_cache = None

    return _manager_instance
