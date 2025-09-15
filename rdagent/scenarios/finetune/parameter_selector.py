"""
Parameter selector for intelligent parameter filtering based on hypothesis.
Selects only relevant parameters for the chosen fine-tuning method.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from rdagent.log import rdagent_logger as logger


class ParameterSelector:
    """Select relevant parameters based on hypothesis and method."""

    def __init__(self, llama_info_path: Path):
        """
        Initialize parameter selector.

        Args:
            llama_info_path: Path to LLaMA Factory info directory
        """
        self.info_path = llama_info_path
        self.params_cache = {}
        self._load_all_params()

    def _load_all_params(self):
        """Load all categorized parameters into cache."""
        try:
            # Load common parameters
            common_dir = self.info_path / "common"
            if common_dir.exists():
                self.params_cache["common"] = {}
                for json_file in common_dir.glob("*.json"):
                    with open(json_file, "r", encoding="utf-8") as f:
                        self.params_cache["common"][json_file.stem] = json.load(f)

            # Load method-specific parameters
            method_dir = self.info_path / "method_specific"
            if method_dir.exists():
                self.params_cache["method_specific"] = {}
                for json_file in method_dir.glob("*.json"):
                    with open(json_file, "r", encoding="utf-8") as f:
                        self.params_cache["method_specific"][json_file.stem] = json.load(f)

            # Load stage-specific parameters
            stage_dir = self.info_path / "stage_specific"
            if stage_dir.exists():
                self.params_cache["stage_specific"] = {}
                for json_file in stage_dir.glob("*.json"):
                    with open(json_file, "r", encoding="utf-8") as f:
                        self.params_cache["stage_specific"][json_file.stem] = json.load(f)

            # Load advanced parameters
            advanced_dir = self.info_path / "advanced"
            if advanced_dir.exists():
                self.params_cache["advanced"] = {}
                for json_file in advanced_dir.glob("*.json"):
                    with open(json_file, "r", encoding="utf-8") as f:
                        self.params_cache["advanced"][json_file.stem] = json.load(f)

        except Exception as e:
            logger.error(f"Failed to load parameters: {e}")
            self.params_cache = {}

    def get_parameters_for_hypothesis(
        self,
        finetune_method: str,
        quantization: Optional[Dict] = None,
        stage: str = "sft",
        include_advanced: bool = False,
    ) -> Dict[str, Dict]:
        """
        Get relevant parameters for a specific hypothesis.

        Args:
            finetune_method: The fine-tuning method (lora, full, freeze, oft)
            quantization: Quantization config dict with 'bit' and 'method'
            stage: Training stage (default: sft)
            include_advanced: Whether to include advanced optimizer params

        Returns:
            Dictionary of categorized parameters
        """
        selected_params = {"common": {}, "method_specific": {}, "stage_specific": {}, "advanced": {}}

        # 1. Always include common parameters
        if "common" in self.params_cache:
            selected_params["common"] = self.params_cache["common"].copy()

        # 2. Add method-specific parameters
        if "method_specific" in self.params_cache:
            # Add parameters for the specific method
            if finetune_method in self.params_cache["method_specific"]:
                selected_params["method_specific"][finetune_method] = self.params_cache["method_specific"][
                    finetune_method
                ]

            # Add quantization parameters if needed
            if quantization and "quantization" in self.params_cache["method_specific"]:
                selected_params["method_specific"]["quantization"] = self.params_cache["method_specific"][
                    "quantization"
                ]

        # 3. Add stage-specific parameters (only SFT for this project)
        if "stage_specific" in self.params_cache and stage in self.params_cache["stage_specific"]:
            selected_params["stage_specific"][stage] = self.params_cache["stage_specific"][stage]

        # 4. Optionally add advanced parameters
        if include_advanced and "advanced" in self.params_cache:
            selected_params["advanced"] = self.params_cache["advanced"].copy()

        return selected_params

    def get_filtered_param_names(self, finetune_method: str, quantization: Optional[Dict] = None) -> Set[str]:
        """
        Get a set of all relevant parameter names for the given method.

        Args:
            finetune_method: The fine-tuning method
            quantization: Quantization config

        Returns:
            Set of parameter names that are relevant
        """
        params = self.get_parameters_for_hypothesis(finetune_method, quantization, include_advanced=True)

        param_names = set()
        for category in params.values():
            for subcategory in category.values():
                if isinstance(subcategory, dict):
                    param_names.update(subcategory.keys())

        return param_names

    def format_params_for_prompt(
        self, finetune_method: str, quantization: Optional[Dict] = None, stage: str = "sft"
    ) -> str:
        """
        Format parameters for LLM prompt in a readable way.

        Args:
            finetune_method: The fine-tuning method
            quantization: Quantization config
            stage: Training stage

        Returns:
            Formatted string for prompt
        """
        params = self.get_parameters_for_hypothesis(finetune_method, quantization, stage)

        lines = [f"Available parameters for {finetune_method} fine-tuning:"]

        # Format common parameters
        if params["common"]:
            lines.append("\n## Common Parameters:")
            for category, category_params in params["common"].items():
                lines.append(f"\n### {category.upper()} Parameters:")
                for param_name, param_info in category_params.items():
                    if "simple_format" in param_info:
                        lines.append(f"- {param_info['simple_format']}")

        # Format method-specific parameters
        if params["method_specific"]:
            lines.append("\n## Method-Specific Parameters:")
            for method, method_params in params["method_specific"].items():
                lines.append(f"\n### {method.upper()} Parameters:")
                for param_name, param_info in method_params.items():
                    if "simple_format" in param_info:
                        lines.append(f"- {param_info['simple_format']}")

        # Format stage-specific parameters
        if params["stage_specific"]:
            lines.append("\n## Stage-Specific Parameters:")
            for stage_name, stage_params in params["stage_specific"].items():
                lines.append(f"\n### {stage_name.upper()} Parameters:")
                for param_name, param_info in stage_params.items():
                    if "simple_format" in param_info:
                        lines.append(f"- {param_info['simple_format']}")

        return "\n".join(lines)

    def validate_param_combination(self, params: Dict[str, any], finetune_method: str) -> List[str]:
        """
        Validate parameter combination for conflicts.

        Args:
            params: Parameters to validate
            finetune_method: The fine-tuning method

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Define incompatible parameter combinations
        incompatible_rules = {
            "lora": {
                "incompatible_with": ["freeze_trainable_layers", "freeze_trainable_modules"],
                "required": ["lora_rank", "lora_target"],
            },
            "freeze": {
                "incompatible_with": ["lora_rank", "lora_alpha", "lora_dropout"],
                "required": ["freeze_trainable_layers"],
            },
            "full": {"incompatible_with": ["lora_rank", "freeze_trainable_layers", "oft_rank"], "required": []},
            "oft": {
                "incompatible_with": ["lora_rank", "freeze_trainable_layers"],
                "required": ["oft_rank", "oft_target"],
            },
        }

        if finetune_method in incompatible_rules:
            rules = incompatible_rules[finetune_method]

            # Check for incompatible parameters
            for param in rules["incompatible_with"]:
                if param in params and params[param] is not None:
                    errors.append(f"Parameter '{param}' is incompatible with {finetune_method} fine-tuning")

            # Check for required parameters
            for param in rules["required"]:
                if param not in params or params[param] is None:
                    errors.append(f"Parameter '{param}' is required for {finetune_method} fine-tuning")

        return errors
