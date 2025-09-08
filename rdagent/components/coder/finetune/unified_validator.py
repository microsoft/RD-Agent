"""
Unified LLM Fine-tuning Configuration Validator

Integrates parameter filtering and completeness checking into a single validation interface.
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
    missing_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class UnifiedLLMConfigValidator:
    """Unified LLM configuration validator with three-tier validation:

    1. Parameter filtering - Remove unsupported parameters
    2. Completeness check - Validate required fields and configuration
    3. Optional micro-batch test - Runtime validation with small dataset
    """

    REQUIRED_FIELDS = ["model_name_or_path", "stage", "do_train", "finetuning_type", "dataset"]
    LORA_FIELDS = ["lora_rank", "lora_alpha"]

    def __init__(self):
        self._supported_params_cache: Optional[Set[str]] = None

    def validate_config_comprehensive(
        self, config_yaml: str, enable_micro_batch_test: bool = False, workspace: Optional[FBWorkspace] = None, env=None
    ) -> ValidationResult:
        """Comprehensive configuration validation with optional micro-batch testing"""
        start_time = time.time()

        try:
            # Tier 1: Parameter filtering
            filtered_config = self._filter_parameters(config_yaml)

            # Tier 2: Completeness validation
            result = self._validate_completeness(filtered_config)

            # Tier 3: Optional micro-batch testing
            if enable_micro_batch_test and workspace and env and result.success:
                dynamic_result = self._run_micro_batch_test(filtered_config, workspace, env)
                result.success = result.success and dynamic_result.success
                result.warnings.extend(dynamic_result.warnings)
                result.errors.extend(dynamic_result.errors)

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(
                success=False,
                filtered_config=config_yaml,
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
        """Get supported parameters from LlamaFactory in Docker environment"""
        if self._supported_params_cache is not None:
            return self._supported_params_cache

        # Fallback parameter set
        fallback_params = {
            "model_name_or_path",
            "stage",
            "do_train",
            "finetuning_type",
            "dataset",
            "lora_rank",
            "lora_alpha",
            "learning_rate",
            "num_train_epochs",
            "max_steps",
            "output_dir",
            "logging_steps",
            "save_steps",
            "warmup_steps",
            "lr_scheduler_type",
            "optim",
            "max_grad_norm",
            "save_strategy",
        }

        try:
            # Execute extraction script in Docker
            workspace = FBWorkspace()
            workspace.inject_files(**{"extract_params.py": (DIRNAME / "extract_params.txt").read_text()})
            result = workspace.run(env=get_ft_env(running_timeout_period=30), entry="python extract_params.py")

            if result.exit_code == 0:
                params = set(json.loads(result.stdout.strip()))
                logger.info(f"Extracted {len(params)} parameters from Docker")
                self._supported_params_cache = params
                return params

        except Exception as e:
            logger.warning(f"Docker extraction failed ({e}), using fallback")

        self._supported_params_cache = fallback_params
        return fallback_params

    def _validate_completeness(self, config_yaml: str) -> ValidationResult:
        """Validate configuration completeness and correctness"""
        result = ValidationResult(success=True, filtered_config=config_yaml)

        try:
            config = yaml.safe_load(config_yaml)
            if not isinstance(config, dict):
                result.success = False
                result.errors.append("Configuration is not a valid dictionary")
                return result

            # Check required fields
            missing_fields = [field for field in self.REQUIRED_FIELDS if field not in config]
            if missing_fields:
                result.success = False
                result.missing_fields = missing_fields
                result.errors.append(f"Missing required fields: {missing_fields}")

            # Check LoRA-specific parameters
            finetuning_type = config.get("finetuning_type", "")
            if finetuning_type in ["lora", "qlora"]:
                missing_lora = [f for f in self.LORA_FIELDS if f not in config]
                if missing_lora:
                    result.warnings.append(f"Missing LoRA fields for {finetuning_type}: {missing_lora}")

            # Check debug mode
            if config.get("max_samples") == 100:
                result.warnings.append("Debug mode detected (max_samples=100)")

            return result

        except yaml.YAMLError as e:
            result.success = False
            result.errors.append(f"YAML parsing error: {e}")
            return result
        except Exception as e:
            result.success = False
            result.errors.append(f"Completeness validation error: {e}")
            return result

    def _run_micro_batch_test(self, config_yaml: str, workspace: FBWorkspace, env) -> ValidationResult:
        """Run micro-batch training test for runtime validation"""
        result = ValidationResult(success=True, filtered_config=config_yaml)

        try:
            # Create micro-batch test configuration
            config = yaml.safe_load(config_yaml)
            if not isinstance(config, dict):
                result.success = False
                result.errors.append("Invalid configuration for micro-batch test")
                return result

            test_config = config.copy()
            test_config.update(
                {
                    "max_samples": 10,
                    "num_train_epochs": 1,
                    "max_steps": 5,
                    "save_steps": 1000,
                    "logging_steps": 1,
                    "warmup_steps": 0,
                    "output_dir": "/workspace/micro_test_output",
                    "overwrite_output_dir": True,
                }
            )

            # Run micro-batch training
            workspace.inject_files(**{"test_train.yaml": yaml.dump(test_config, default_flow_style=False)})
            training_result = workspace.run(env=env, entry="timeout 300 llamafactory-cli train test_train.yaml")

            # Check results
            progress_indicators = ["train_loss", "Training:", "Epoch", "loss:", "step"]
            has_progress = any(ind.lower() in training_result.stdout.lower() for ind in progress_indicators)

            if training_result.exit_code == 0 and has_progress:
                logger.info("Micro-batch test passed")
            else:
                result.success = False
                result.errors.append(f"Micro-batch test failed (exit_code={training_result.exit_code})")

            return result

        except Exception as e:
            result.success = False
            result.errors.append(f"Micro-batch test exception: {str(e)}")
            return result

    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generate validation report"""
        status = "PASSED" if result.success else "FAILED"
        report = f"=== LLM Configuration Validation Report ===\n"
        report += f"Status: {status} (took {result.execution_time:.2f}s)\n"

        if result.missing_fields:
            report += f"Missing fields: {result.missing_fields}\n"
        if result.warnings:
            report += f"Warnings: {'; '.join(result.warnings)}\n"
        if result.errors:
            report += f"Errors: {'; '.join(result.errors)}\n"

        return report


def create_unified_validator() -> UnifiedLLMConfigValidator:
    """Create unified validator instance"""
    return UnifiedLLMConfigValidator()
