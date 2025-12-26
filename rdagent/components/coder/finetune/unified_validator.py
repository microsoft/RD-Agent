"""
Simplified LLM Fine-tuning Configuration Validator

Two-step validation:
1. Parameter filtering - Remove unsupported parameters
2. Micro-batch testing - Runtime validation with small dataset
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from rdagent.components.coder.finetune.conf import (
    FT_DEBUG_YAML_FILE_NAME,
    get_ft_env,
    get_workspace_prefix,
)
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.scen.llama_factory_manager import LLaMAFactory_manager

DIRNAME = Path(__file__).absolute().resolve().parent


@dataclass
class ValidationResult:
    """Configuration validation result"""

    success: bool
    filtered_config: str
    execution_output: str = ""  # Parsed/summarized output for LLM
    raw_stdout: str = ""  # Full raw stdout for UI display
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

        # Step 1: Parameter filtering
        filtered_config, removed_params = self._filter_parameters(config_yaml)

        # Step 2: Micro-batch testing (validates everything at runtime)
        result = self._run_micro_batch_test(filtered_config, workspace, env)
        result.execution_time = time.time() - start_time

        # Add filtered params info to execution_output for agent learning
        if removed_params:
            filter_info = (
                f"\n\n[Filtered Parameters] {len(removed_params)} unsupported params removed: {removed_params}"
            )
            result.execution_output += filter_info

        return result

    def _filter_parameters(self, config_yaml: str) -> tuple[str, List[str]]:
        """Filter configuration parameters to only include supported ones.

        Returns:
            tuple: (filtered_yaml, removed_params_list)
        """
        config_dict = yaml.safe_load(config_yaml)
        if not isinstance(config_dict, dict):
            return config_yaml, []

        supported_params = self._get_supported_parameters()

        filtered_config = {}
        removed_params = []
        for k, v in config_dict.items():
            if k in supported_params:
                filtered_config[k] = v
            else:
                removed_params.append(k)

        if removed_params:
            logger.info(f"Filtered out {len(removed_params)} unsupported parameters: {removed_params}")

        return yaml.dump(filtered_config, default_flow_style=False, sort_keys=False), removed_params

    def _get_supported_parameters(self) -> Set[str]:
        """Get supported parameters from LlamaFactory Manager"""
        if self._supported_params_cache is not None:
            return self._supported_params_cache

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

    def _parse_execution_log(self, stdout: str, exit_code: int) -> str:
        """Parse execution log and extract key information for LLM evaluation.

        Reduces log from ~36k tokens to ~500 tokens by extracting only:
        - Status and exit code
        - Error messages (if any)
        - Training metrics (if successful)
        - Warnings (limited)
        """
        result = {
            "status": "success" if exit_code == 0 else "failed",
            "exit_code": exit_code,
        }

        # 1. Extract error information (highest priority)
        # Strategy: extract rank0's error block (each line prefixed with [rank0]:)
        error_text = None

        # Method A: Extract [rank0]: prefixed lines and reconstruct traceback
        rank0_lines = re.findall(r"\[rank0\]:[^\n]+", stdout)
        if rank0_lines:
            rank0_block = "\n".join(line.replace("[rank0]: ", "").replace("[rank0]:", "") for line in rank0_lines)
            # Find traceback in rank0 block
            tb_match = re.search(
                r"Traceback \(most recent call last\):.*?(?:Error|Exception):[^\n]+", rank0_block, re.DOTALL
            )
            if tb_match:
                error_text = tb_match.group(0)

        # Method B: Fallback to generic traceback (no rank prefix)
        # Use findall to get ALL tracebacks, then keep the first one (root cause)
        if not error_text:
            all_tracebacks = re.findall(
                r"Traceback \(most recent call last\):.*?(?:Error|Exception):[^\n]+", stdout, re.DOTALL
            )
            if all_tracebacks:
                # First traceback is usually the root cause
                error_text = all_tracebacks[0]
                if len(all_tracebacks) > 1:
                    error_text += f"\n\n[Note: {len(all_tracebacks)} total errors, showing root cause]"

        if error_text:
            # Limit length but keep from the END (actual error type/message is at the end of traceback)
            result["error"] = error_text[-4000:] if len(error_text) > 4000 else error_text

        # 2. Extract training information
        if "Running training" in stdout:
            result["training_started"] = True

            # Extract training config
            num_examples = re.search(r"Num examples\s*=\s*(\d+)", stdout)
            num_epochs = re.search(r"Num Epochs\s*=\s*(\d+)", stdout)
            if num_examples:
                result["num_examples"] = int(num_examples.group(1))
            if num_epochs:
                result["num_epochs"] = int(num_epochs.group(1))

            # Extract final metrics (JSON format from trainer output)
            final_metrics = re.search(r"\{'train_runtime':[^}]+\}", stdout)
            if final_metrics:
                try:
                    metrics = eval(final_metrics.group(0))  # Safe: only numbers and strings
                    result["final_metrics"] = {
                        "train_loss": metrics.get("train_loss"),
                        "train_runtime": metrics.get("train_runtime"),
                        "train_samples_per_second": metrics.get("train_samples_per_second"),
                    }
                except Exception:
                    pass

            # Check completion
            if "Training completed" in stdout:
                result["completed"] = True

        # 3. Extract warnings (limit to 20)
        warnings = re.findall(r"\[WARNING[^\]]*\][^\n]+", stdout)
        if warnings:
            result["warnings"] = list(set(warnings))[:20]

        # 4. Fallback: if parsing failed, include truncated raw log
        if not result.get("error") and not result.get("training_started"):
            result["raw_log_tail"] = stdout[-2000:] if len(stdout) > 2000 else stdout

        return json.dumps(result, indent=2, ensure_ascii=False)

    def _run_micro_batch_test(self, config_yaml: str, workspace: FBWorkspace, env) -> ValidationResult:
        """Run micro-batch training test for runtime validation"""
        result = ValidationResult(success=True, filtered_config=config_yaml)
        ws_prefix = get_workspace_prefix(env)

        # Create micro-batch test configuration
        config = yaml.safe_load(config_yaml)
        if not isinstance(config, dict):
            result.success = False
            result.execution_output = "Invalid YAML configuration"
            result.errors.append("Invalid configuration for micro-batch test")
            return result

        test_config = config.copy()
        # NOTE: save_steps/logging_steps/warmup_steps not overridden - debug data.json only has ~10 samples,
        # so total training steps are naturally small and these optimizations are unnecessary
        test_config.update(
            {
                "num_train_epochs": 1,
                "output_dir": f"{ws_prefix}/micro_test_output",
                "overwrite_output_dir": True,
                "report_to": "none",  # Disable all reporting (tensorboard, wandb, etc.)
                "do_eval": False,  # Disable evaluation in micro-batch test (insufficient samples for val split)
                "eval_strategy": "no",  # Explicitly disable evaluation
                "load_best_model_at_end": False,  # Cannot load best model without evaluation
                "tokenized_path": f"{ws_prefix}/micro_test_cache",  # Use writable workspace instead of read-only /assets
            }
        )

        # Run micro-batch training
        workspace.inject_files(**{FT_DEBUG_YAML_FILE_NAME: yaml.dump(test_config, default_flow_style=False)})
        training_result = workspace.run(
            env=env,
            entry=f"llamafactory-cli train {FT_DEBUG_YAML_FILE_NAME}",
        )

        # Remove micro-batch test files
        workspace.remove_files([FT_DEBUG_YAML_FILE_NAME])

        # Parse and store structured execution output (reduces ~36k tokens to ~500)
        raw_stdout = training_result.stdout if training_result.stdout else ""
        result.raw_stdout = raw_stdout  # Keep full log for UI
        result.execution_output = self._parse_execution_log(raw_stdout, training_result.exit_code)

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
