"""
LLaMA Factory Parameter Validator

Uses official LLaMA Factory interfaces to validate and filter configuration parameters.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

import yaml

logger = logging.getLogger(__name__)


class LlamaFactoryParameterValidator:
    """Validates and filters LLaMA Factory configuration parameters using official interfaces."""

    def __init__(self):
        self._supported_params_cache: Optional[Set[str]] = None

    def get_supported_parameters(self) -> Set[str]:
        """Get supported parameters from LLaMA Factory's argument classes."""
        if self._supported_params_cache is not None:
            return self._supported_params_cache

        supported_params = set()

        try:
            # Import LLaMA Factory's argument classes directly
            from llamafactory.hparams.data_args import DataArguments
            from llamafactory.hparams.finetuning_args import FinetuningArguments
            from llamafactory.hparams.model_args import ModelArguments
            from transformers import TrainingArguments

            # Get fields from each dataclass
            for arg_class in [
                DataArguments,
                ModelArguments,
                FinetuningArguments,
                TrainingArguments,
            ]:
                if hasattr(arg_class, "__dataclass_fields__"):
                    class_params = arg_class.__dataclass_fields__.keys()
                    supported_params.update(class_params)
                    logger.debug(f"Added {len(class_params)} parameters from {arg_class.__name__}")

            # Also try to get generating args if available
            try:
                from llamafactory.hparams.generating_args import GeneratingArguments

                if hasattr(GeneratingArguments, "__dataclass_fields__"):
                    gen_params = GeneratingArguments.__dataclass_fields__.keys()
                    supported_params.update(gen_params)
                    logger.debug(f"Added {len(gen_params)} parameters from GeneratingArguments")
            except ImportError:
                logger.debug("GeneratingArguments not available")

            logger.info(f"Successfully loaded {len(supported_params)} supported parameters from LLaMA Factory")

        except ImportError as e:
            logger.warning(f"Could not import LLaMA Factory argument classes: {e}")
            # Fallback to a known good set of parameters
            supported_params = self._get_fallback_parameters()
            logger.info(f"Using fallback parameter set with {len(supported_params)} parameters")

        self._supported_params_cache = supported_params
        return supported_params

    def _get_fallback_parameters(self) -> Set[str]:
        """Fallback parameter set when LLaMA Factory imports fail."""
        return {
            # Model parameters
            "model_name_or_path",
            "model_revision",
            "quantization_bit",
            "rope_scaling",
            "trust_remote_code",
            "use_fast_tokenizer",
            "resize_vocab",
            "split_special_tokens",
            # Data parameters
            "dataset",
            "template",
            "cutoff_len",
            "train_on_prompt",
            "mask_history",
            "max_source_length",
            "max_target_length",
            "preprocessing_num_workers",
            "max_samples",
            "val_size",
            "test_size",
            # Training parameters
            "stage",
            "do_train",
            "do_eval",
            "do_predict",
            "output_dir",
            "overwrite_output_dir",
            "num_train_epochs",
            "max_steps",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "max_grad_norm",
            "warmup_steps",
            "warmup_ratio",
            "lr_scheduler_type",
            "logging_steps",
            "save_steps",
            "eval_steps",
            "save_total_limit",
            "metric_for_best_model",
            "greater_is_better",
            "load_best_model_at_end",
            "evaluation_strategy",
            "save_strategy",
            "auto_find_batch_size",
            "dataloader_drop_last",
            "eval_do_concat_batches",
            "fp16",
            "bf16",
            "tf32",
            "dataloader_pin_memory",
            "dataloader_num_workers",
            "past_index",
            "run_name",
            "disable_tqdm",
            "remove_unused_columns",
            "label_names",
            "report_to",
            "optim",
            "optim_args",
            "adafactor",
            "group_by_length",
            "length_column_name",
            "ddp_timeout",
            "ddp_backend",
            "ddp_bucket_cap_mb",
            "ddp_broadcast_buffers",
            "dataloader_persistent_workers",
            "skip_memory_metrics",
            "use_legacy_prediction_loop",
            "push_to_hub",
            "resume_from_checkpoint",
            "hub_model_id",
            "hub_strategy",
            "hub_token",
            "hub_private_repo",
            "hub_always_push",
            "gradient_checkpointing",
            "include_inputs_for_metrics",
            "eval_accumulation_steps",
            "eval_delay",
            "torch_compile",
            "torch_compile_backend",
            "torch_compile_mode",
            "dispatch_batches",
            "split_batches",
            "include_tokens_per_second",
            # LoRA parameters
            "finetuning_type",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_target",
            "additional_target",
            "lora_bias",
            "use_rslora",
            "use_dora",
            "pissa_init",
            "pissa_iter",
            "pissa_convert",
            "create_new_adapter",
            "freeze_trainable_layers",
            "freeze_trainable_modules",
            "freeze_extra_modules",
            # Generation parameters
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "num_beams",
            "max_length",
            "max_new_tokens",
            "repetition_penalty",
            "length_penalty",
            "default_system",
            # Evaluation parameters
            "eval_num_beams",
            "predict_with_generate",
            # Other parameters
            "seed",
            "data_seed",
            "jit_mode_eval",
            "use_ipex",
            "bf16_full_eval",
            "fp16_full_eval",
            "tf32_full_eval",
            "local_rank",
            "deepspeed",
            "label_smoothing_factor",
            "debug",
            "sharded_ddp",
            "fsdp",
            "fsdp_config",
            "fsdp_min_num_params",
            "fsdp_transformer_layer_cls_to_wrap",
            "accelerator_config",
            "deepspeed_plugin",
            "fsdp_plugin",
        }

    def validate_config_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and filter configuration dictionary."""
        supported_params = self.get_supported_parameters()

        valid_config = {}
        invalid_params = []

        for key, value in config_dict.items():
            if key in supported_params:
                valid_config[key] = value
            else:
                invalid_params.append(key)
                logger.warning(f"Removing unsupported parameter: {key}")

        if invalid_params:
            logger.info(f"Filtered out {len(invalid_params)} unsupported parameters: {invalid_params}")

        return valid_config

    def validate_yaml_config(self, yaml_content: str) -> str:
        """Validate and filter YAML configuration content."""
        try:
            config_dict = yaml.safe_load(yaml_content)
            if not isinstance(config_dict, dict):
                logger.warning("YAML content is not a dictionary")
                return yaml_content

            valid_config = self.validate_config_dict(config_dict)
            return yaml.dump(valid_config, default_flow_style=False, sort_keys=False)

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML: {e}")
            return yaml_content

    def validate_config_file(self, config_path: Path) -> bool:
        """Validate and update configuration file in place."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            valid_content = self.validate_yaml_config(original_content)

            if valid_content != original_content:
                # Backup original file
                backup_path = config_path.with_suffix(".yaml.backup")
                with open(backup_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                logger.info(f"Created backup: {backup_path}")

                # Write validated content
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(valid_content)
                logger.info(f"Updated configuration file: {config_path}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error validating config file {config_path}: {e}")
            return False


def create_parameter_validator() -> LlamaFactoryParameterValidator:
    """Create a parameter validator instance."""
    return LlamaFactoryParameterValidator()
