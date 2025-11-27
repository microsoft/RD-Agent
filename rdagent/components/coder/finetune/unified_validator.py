"""
Simplified LLM Fine-tuning Configuration Validator

Two-step validation:
1. Parameter filtering - Remove unsupported parameters
2. Micro-batch testing - Runtime validation with small dataset
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from rdagent.components.coder.finetune.conf import FT_DEBUG_YAML_FILE_NAME, get_ft_env
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.scen.llama_factory_manager import LLaMAFactory_manager

DIRNAME = Path(__file__).absolute().resolve().parent


@dataclass
class ValidationResult:
    """Configuration validation result"""

    success: bool
    filtered_config: str
    execution_output: str = ""  # stdout/stderr from micro-batch test
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class LLMConfigValidator:
    """LLM configuration validator with two-step validation:

    1. Parameter filtering - Remove unsupported parameters
    2. Micro-batch test - Runtime validation with small dataset

    The micro-batch test inherently validates completeness, so no separate completeness check is needed.
    """

    def __init__(self):
        self._supported_params_cache: Optional[Set[str]] = None

    def validate_and_test(self, config_yaml: str, workspace: FBWorkspace, env) -> ValidationResult:
        """Two-step validation: parameter filtering + micro-batch testing"""
        start_time = time.time()

        try:
            # Step 1: Parameter filtering
            filtered_config = self._filter_parameters(config_yaml)

            # Step 2: Micro-batch testing (validates everything at runtime)
            result = self._run_micro_batch_test(filtered_config, workspace, env)
            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(
                success=False,
                filtered_config=config_yaml,
                execution_output=f"Validation exception: {str(e)}",
                errors=[f"Validation exception: {str(e)}"],
                execution_time=time.time() - start_time,
            )

    def _filter_parameters(self, config_yaml: str) -> str:
        """Filter configuration parameters to only include supported ones"""
        try:
            config_dict = yaml.safe_load(config_yaml)
            if not isinstance(config_dict, dict):
                return config_yaml

            supported_params = self._get_supported_parameters()

            # filter parameters
            filtered_config = {}
            removed_params = []
            for k, v in config_dict.items():
                if k in supported_params:
                    filtered_config[k] = v
                else:
                    removed_params.append(k)

            removed_count = len(removed_params)
            if removed_count > 0:
                logger.info(f"Filtered out {removed_count} unsupported parameters: {removed_params}")

            return yaml.dump(filtered_config, default_flow_style=False, sort_keys=False)

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error during filtering: {e}")
            return config_yaml

    def _get_supported_parameters(self) -> Set[str]:
        """Get supported parameters from LlamaFactory Manager"""
        if self._supported_params_cache is not None:
            return self._supported_params_cache

        try:
            all_params = LLaMAFactory_manager.get_parameters()

            # Extract all parameter names from all parameter types (including nested structures)
            supported_params = set()
            for param_type, params_dict in all_params.items():
                if isinstance(params_dict, dict):
                    # Recursively extract parameter names from nested dictionaries
                    for key, value in params_dict.items():
                        if isinstance(value, dict) and "name" in value:
                            # This is a parameter definition with metadata
                            supported_params.add(key)
                        elif isinstance(value, dict):
                            # This is a nested category (e.g., BaseModelArguments, LoraArguments)
                            # Extract parameter names from the nested structure
                            for nested_key, nested_value in value.items():
                                if isinstance(nested_value, dict) and "name" in nested_value:
                                    supported_params.add(nested_key)

            if not supported_params:
                raise RuntimeError("No parameters found in LlamaFactory Manager")

            logger.info(f"Loaded {len(supported_params)} parameters from LlamaFactory Manager")
            self._supported_params_cache = supported_params
            return supported_params

        except Exception as e:
            logger.error(f"Failed to load parameters from LlamaFactory Manager: {e}")
            raise RuntimeError(f"Unable to get supported parameters from LlamaFactory: {e}") from e

    def _run_micro_batch_test(self, config_yaml: str, workspace: FBWorkspace, env) -> ValidationResult:
        """Run micro-batch training test for runtime validation"""
        result = ValidationResult(success=True, filtered_config=config_yaml)

        try:
            # Create micro-batch test configuration
            config = yaml.safe_load(config_yaml)
            if not isinstance(config, dict):
                result.success = False
                result.execution_output = "Invalid YAML configuration"
                result.errors.append("Invalid configuration for micro-batch test")
                return result

            test_config = config.copy()
            test_config.update(
                {
                    "max_samples": 4,
                    "num_train_epochs": 1,
                    "max_steps": 2,
                    "save_steps": 1000,
                    "logging_steps": 1,
                    "warmup_steps": 0,
                    "output_dir": "/workspace/micro_test_output",
                    "overwrite_output_dir": True,
                    "report_to": "none",  # Disable all reporting (tensorboard, wandb, etc.)
                    "do_eval": False,  # Disable evaluation in micro-batch test (insufficient samples for val split)
                    "eval_strategy": "no",  # Explicitly disable evaluation
                    "load_best_model_at_end": False,  # Cannot load best model without evaluation
                    "tokenized_path": "/workspace/micro_test_cache",  # Use writable workspace instead of read-only /assets
                }
            )

            # Run micro-batch training
            workspace.inject_files(**{FT_DEBUG_YAML_FILE_NAME: yaml.dump(test_config, default_flow_style=False)})
            training_result = workspace.run(
                env=env, entry=f"timeout 300 llamafactory-cli train {FT_DEBUG_YAML_FILE_NAME}"
            )

            # Remove micro-batch test files
            workspace.remove_files([FT_DEBUG_YAML_FILE_NAME])

            # Store all execution output
            result.execution_output = training_result.stdout if training_result.stdout else ""

            # Check results
            progress_indicators = ["train_loss", "Training:", "Epoch", "loss:", "step"]
            has_progress = any(ind.lower() in training_result.stdout.lower() for ind in progress_indicators)

            if training_result.exit_code == 0 and has_progress:
                logger.info("Micro-batch test passed")
                result.success = True
            else:
                result.success = False
                result.errors.append(f"Micro-batch test failed (exit_code={training_result.exit_code})")

            return result

        except Exception as e:
            result.success = False
            result.execution_output = str(e)
            result.errors.append(f"Micro-batch test exception: {str(e)}")
            return result
