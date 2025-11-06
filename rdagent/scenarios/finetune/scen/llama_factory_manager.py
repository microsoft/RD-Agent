"""
Streamlined LLaMA Factory manager for parameter extraction.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger

UPDATE_LLAMA_FACTORY_SCRIPT_NAME = "update_llama_factory_extract_parameters.py"


class LLaMAFactoryManager:
    """Manager for LLaMA Factory parameter extraction and caching."""

    def __init__(self):
        """Initialize the manager instance."""
        base_path = FT_RD_SETTING.file_path
        self.cache_dir = Path(base_path) / ".llama_factory_info"
        self._info_cache: Optional[Dict] = None
        self.update_llama_factory = FT_RD_SETTING.update_llama_factory

    def extract_info_from_docker(self) -> Dict:
        """Extract LLaMA Factory information from Docker environment."""
        if self.update_llama_factory or not self.cache_dir.exists() or not any(self.cache_dir.iterdir()):
            logger.info("Update & Extract LLaMA Factory parameters from Docker")
            # Prepare extraction script
            workspace = FBWorkspace()
            script_path = Path(__file__).parent / "docker_scripts" / UPDATE_LLAMA_FACTORY_SCRIPT_NAME
            workspace.inject_files(**{UPDATE_LLAMA_FACTORY_SCRIPT_NAME: script_path.read_text()})

            # Setup cache directory and Docker volumes
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            volumes = {str(self.cache_dir): {"bind": "/workspace/.llama_factory_info", "mode": "rw"}}

            # Run extraction
            result = workspace.run(
                env=get_ft_env(extra_volumes=volumes, running_timeout_period=120, enable_cache=False),
                entry=f"python {UPDATE_LLAMA_FACTORY_SCRIPT_NAME}",
            )

            if result.exit_code != 0:
                raise RuntimeError(f"Parameter extraction failed: {result.stdout}")

        else:
            logger.info("Skip updating LLaMA Factory, using local cache")

        # Load the extracted data
        self._info_cache = self._load_extracted_data()
        if not self._info_cache:
            raise RuntimeError("Failed to load extracted LLaMA Factory information")

        logger.info("Successfully extracted LLaMA Factory parameters")
        return self._info_cache

    def _load_extracted_data(self) -> Dict:
        """Load extracted information from flat file structure."""
        data = {}

        # Load constants
        constants_file = self.cache_dir / "constants.json"
        if constants_file.exists():
            with open(constants_file, encoding="utf-8") as f:
                data.update(json.load(f))

        # Load parameters
        parameters_file = self.cache_dir / "parameters.json"
        if parameters_file.exists():
            with open(parameters_file, encoding="utf-8") as f:
                data["parameters"] = json.load(f)

        return data

    def get_info(self) -> Dict:
        """Get complete LLaMA Factory information, extracting on first call."""
        if self._info_cache is None:
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
        """Available HuggingFace models."""
        supported_models = self.get_info().get("supported_models", {})
        hf_model_set = set()
        for hf_model in supported_models.values():
            if isinstance(hf_model, str):
                hf_model_set.add(hf_model)
        return list(hf_model_set)

    @property
    def peft_methods(self) -> List[str]:
        """Available PEFT methods, dynamically filtered from available methods."""
        known_peft = {"lora", "qlora", "adalora"}
        return [m for m in self.methods if m in known_peft]

    @property
    def training_stages(self) -> Dict[str, str]:
        """Training stage mapping."""
        return self.get_info().get("training_stages", {})

    @property
    def templates(self) -> List[str]:
        """Available chat templates."""
        return self.get_info().get("templates", [])

    def get_template_for_model(self, model_name: str) -> Optional[str]:
        """Get template for model. Returns None to let LlamaFactory auto-detect.

        Args:
            model_name: Model name (e.g., "Qwen/Qwen2.5-1.5B-Instruct")

        Returns:
            None - LlamaFactory will automatically detect the appropriate template
        """
        return None

    def is_peft_method(self, method: str) -> bool:
        """Check if the given method is a PEFT method."""
        return method in self.peft_methods

    def get_parameters(self, param_type: Optional[str] = None) -> Dict:
        """Get parameters by type or all parameters."""
        params = self.get_info().get("parameters", {})
        if param_type:
            return params.get(param_type, {})
        return params

    def format_method_params(self, method: str) -> str:
        """Format parameters for a specific method as a readable string."""
        lines = [f"Parameters for {method} fine-tuning:"]
        all_params = self.get_parameters()

        # Format all parameter categories
        for param_type in ["model", "data", "training", "finetuning"]:
            if param_type not in all_params:
                continue

            lines.append(f"\n### {param_type.upper()} Parameters:")
            type_params = all_params[param_type]

            # Filter method-specific parameters for finetuning category
            if param_type == "finetuning":
                if method == "lora":
                    type_params = {k: v for k, v in type_params.items() if "lora" in k.lower()}
                elif method == "freeze":
                    type_params = {k: v for k, v in type_params.items() if "freeze" in k.lower()}

            for param_name, param_info in type_params.items():
                if isinstance(param_info, dict) and "help" in param_info:
                    help_text = param_info["help"][:100]  # Truncate long help text
                    lines.append(f"- {param_name}: {help_text}")

        return "\n".join(lines)


# Module-level singleton instance
_manager_instance: Optional[LLaMAFactoryManager] = None


def get_llama_factory_manager() -> LLaMAFactoryManager:
    """Get the singleton LLaMAFactoryManager instance."""
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = LLaMAFactoryManager()

    return _manager_instance
