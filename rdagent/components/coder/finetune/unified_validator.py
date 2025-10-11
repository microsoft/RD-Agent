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

from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger

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

    def __init__(self, llama_factory_manager=None):
        self._supported_params_cache: Optional[Set[str]] = None
        self.llama_factory_manager = llama_factory_manager

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

        if self.llama_factory_manager is None:
            raise RuntimeError("LlamaFactory Manager not provided to validator")

        try:
            all_params = self.llama_factory_manager.get_parameters()

            # Extract all parameter names from all parameter types
            supported_params = set()
            for param_type, params_dict in all_params.items():
                if isinstance(params_dict, dict):
                    supported_params.update(params_dict.keys())

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
                }
            )

            # Run micro-batch training
            workspace.inject_files(**{"test_train.yaml": yaml.dump(test_config, default_flow_style=False)})
            training_result = workspace.run(env=env, entry="timeout 300 llamafactory-cli train test_train.yaml")

            # Store execution output (last 2000 chars to keep it manageable)
            result.execution_output = training_result.stdout[-2000:] if training_result.stdout else ""

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

    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generate simplified validation report"""
        status = "PASSED" if result.success else "FAILED"
        report = f"=== LLM Configuration Validation Report ===\n"
        report += f"Status: {status} (took {result.execution_time:.2f}s)\n"

        if result.errors:
            report += f"Errors: {'; '.join(result.errors)}\n"

        if result.execution_output:
            report += f"\n--- Micro-batch Test Output (last 2000 chars) ---\n"
            report += result.execution_output

        return report


def create_unified_validator(llama_factory_manager=None) -> LLMConfigValidator:
    """Create simplified validator instance."""
    if llama_factory_manager is None:
        # Lazy import to avoid circular dependency
        from rdagent.scenarios.finetune.scen.llama_factory_manager import (
            get_llama_factory_manager,
        )

        llama_factory_manager = get_llama_factory_manager()
    return LLMConfigValidator(llama_factory_manager)
