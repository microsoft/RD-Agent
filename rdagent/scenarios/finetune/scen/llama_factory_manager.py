"""
LLaMA Factory manager for parameter extraction and caching.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger


class LLaMAFactoryManager:
    """Manager for LLaMA Factory information extraction and caching."""

    def __init__(self):
        """Initialize the manager instance."""
        base_path = FT_RD_SETTING.file_path
        self.cache_dir = Path(base_path) / ".llama_factory_info"
        self._info_cache: Optional[Dict] = None

    def get_docker_commit_hash(self) -> Optional[str]:
        """Get LLaMA Factory git commit hash from Docker environment."""
        logger.info("Getting LLaMA Factory commit hash from Docker")

        # Use independent script to get commit hash
        workspace = FBWorkspace()
        script_path = Path(__file__).parent / "docker_scripts" / "get_commit.py"
        workspace.inject_files(**{"get_commit.py": script_path.read_text()})

        # Run in Docker
        result = workspace.run(
            env=get_ft_env(running_timeout_period=30),
            entry="python get_commit.py",
        )

        if result.exit_code != 0:
            logger.warning(f"Failed to get commit hash: {result.stdout}")
            return None

        return result.stdout.strip()

    def extract_info_from_docker(self) -> Dict:
        """Extract LLaMA Factory information from Docker environment."""
        logger.info("Extracting LLaMA Factory information from Docker environment")

        # Prepare extraction script
        workspace = FBWorkspace()
        script_path = Path(__file__).parent / "docker_scripts" / "extract_llama_factory.py"
        workspace.inject_files(**{"extract_script.py": script_path.read_text()})

        # Setup cache directory and Docker volumes
        # Clear existing cache directory to ensure fresh extraction
        if self.cache_dir.exists():
            logger.info("Clearing existing LLaMA Factory cache directory")
            shutil.rmtree(self.cache_dir)
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

    def get_info(self) -> Dict:
        """Get complete LLaMA Factory information.

        On first call, checks if Docker's LLaMA Factory commit matches cached version.
        If different or no cache exists, extracts fresh information from Docker.
        """
        if self._info_cache is None:
            # Get current Docker commit hash
            docker_commit = self.get_docker_commit_hash()

            # Try loading from cache
            cached_data = self._load_categorized_structure()
            cached_commit = cached_data.get("llama_factory_commit") if cached_data else None

            # If commits match, use cached data; otherwise extract fresh info
            if docker_commit and cached_commit and docker_commit == cached_commit:
                logger.info(f"Loaded LLaMA Factory info from cache (commit: {docker_commit})")
                self._info_cache = cached_data
            else:
                if docker_commit and cached_commit:
                    logger.info(
                        f"LLaMA Factory commit changed from {cached_commit} to {docker_commit}, extracting fresh info"
                    )
                else:
                    logger.info("Cache invalid or missing, extracting fresh info")
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
    def hf_models(self) -> List[str]:
        """Available HuggingFace models (values from supported_models dict)."""
        supported_models = self.get_info().get("supported_models", {})
        # Extract unique HF model names from the values
        hf_model_set = set()
        for hf_model in supported_models.values():
            if isinstance(hf_model, str):
                hf_model_set.add(hf_model)
        return list(hf_model_set)

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

    def get_metadata_info(self) -> Dict:
        """Get metadata information including commit hash."""
        info = self.get_info()

        # Extract metadata with backward compatibility
        metadata = {
            "has_metadata": bool(info.get("llama_factory_commit")),
            "commit_sha": info.get("llama_factory_commit", "unknown"),
            "timestamp": info.get("timestamp"),
            "version": info.get("version"),
            "last_updated": info.get("last_updated"),
        }

        return metadata

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


def get_llama_factory_manager() -> LLaMAFactoryManager:
    """Get the singleton LLaMAFactoryManager instance."""
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = LLaMAFactoryManager()

    return _manager_instance
