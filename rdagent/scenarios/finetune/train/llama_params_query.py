"""
LLaMA Factory Parameter Query Module

This module provides functionality to query LLaMA Factory parameters based on the fine-tuning method.
TODO: fetch parameters from llama factory dynamically
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LLaMAFactoryParamsQuery:
    """Query LLaMA Factory parameters for different fine-tuning methods."""

    def __init__(self, llama_factory_path: Optional[Path] = None):
        """Initialize the parameter query with LLaMA Factory repository path."""
        if llama_factory_path is None:
            llama_factory_path = Path.cwd() / "git_ignore_folder" / "LLaMA-Factory"
        self.llama_factory_path = llama_factory_path
        self.params_path = llama_factory_path / "src" / "llamafactory" / "hparams"

    def _extract_dataclass_fields(self, file_content: str, class_name: str) -> List[Tuple[str, str, str]]:
        """Extract field definitions from a dataclass.

        Returns:
            List of tuples: (field_name, field_type, help_text)
        """
        try:
            tree = ast.parse(file_content)
            fields = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            field_name = item.target.id
                            field_type = (
                                ast.unparse(item.annotation) if hasattr(ast, "unparse") else str(item.annotation)
                            )

                            # Extract help text from metadata
                            help_text = ""
                            if (
                                isinstance(item.value, ast.Call)
                                and hasattr(item.value.func, "id")
                                and item.value.func.id == "field"
                            ):
                                for keyword in item.value.keywords:
                                    if keyword.arg == "metadata":
                                        if isinstance(keyword.value, ast.Dict):
                                            for key, value in zip(keyword.value.keys, keyword.value.values):
                                                if isinstance(key, ast.Constant) and key.value == "help":
                                                    if isinstance(value, ast.Constant):
                                                        help_text = value.value

                            fields.append((field_name, field_type, help_text))

            return fields
        except Exception as e:
            logger.error(f"Failed to parse AST: {e}")
            return []

    def get_common_training_params(self) -> Dict[str, str]:
        """Get common training parameters applicable to all methods."""
        return {
            "model_name_or_path": "Base model identifier from HuggingFace or local path",
            "dataset": "Dataset name (must match dataset_info.json)",
            "dataset_dir": "Directory containing dataset_info.json",
            "template": "Chat template for the model (e.g., 'qwen', 'llama3', 'alpaca')",
            "stage": "Training stage: 'pt' (pretraining), 'sft' (supervised finetuning), 'rm' (reward modeling), 'ppo' (PPO training), 'dpo' (DPO training)",
            "do_train": "Whether to run training (set to true)",
            "finetuning_type": "Fine-tuning method: 'lora', 'qlora', 'full', 'freeze'",
            "output_dir": "Directory to save model checkpoints and logs",
            "per_device_train_batch_size": "Training batch size per GPU/CPU",
            "gradient_accumulation_steps": "Number of steps to accumulate gradients",
            "learning_rate": "Initial learning rate",
            "num_train_epochs": "Total number of training epochs",
            "max_samples": "Maximum number of samples to use (for debugging)",
            "logging_steps": "Log every N steps",
            "save_steps": "Save checkpoint every N steps",
            "warmup_steps": "Number of warmup steps",
            "lr_scheduler_type": "Learning rate scheduler type",
            "cutoff_len": "Maximum sequence length",
            "overwrite_output_dir": "Overwrite the output directory",
            "fp16": "Use FP16 training",
            "bf16": "Use BF16 training (recommended for newer GPUs)",
        }

    def get_lora_params(self) -> Dict[str, str]:
        """Get LoRA-specific parameters."""
        try:
            file_path = self.params_path / "finetuning_args.py"
            if file_path.exists():
                content = file_path.read_text()
                fields = self._extract_dataclass_fields(content, "LoraArguments")

                # Convert to dict with simplified descriptions
                params = {}
                for field_name, field_type, help_text in fields:
                    if help_text:
                        params[field_name] = help_text.replace("\n", " ").strip()
                    else:
                        params[field_name] = f"Type: {field_type}"

                # Add commonly used LoRA parameters with clear descriptions
                params.update(
                    {
                        "lora_rank": "The intrinsic dimension for LoRA (commonly 8, 16, or 32)",
                        "lora_alpha": "The scale factor for LoRA (typically lora_rank * 2)",
                        "lora_dropout": "Dropout probability for LoRA layers (commonly 0.05 or 0.1)",
                        "lora_target": "Target modules to apply LoRA. Use 'all' for all linear layers or specify modules like 'q_proj,v_proj'",
                    }
                )

                return params
        except Exception as e:
            logger.error(f"Failed to get LoRA params: {e}")

        # Fallback parameters
        return {
            "lora_rank": "LoRA rank (dimension of adaptation)",
            "lora_alpha": "LoRA scaling parameter",
            "lora_dropout": "LoRA dropout rate",
            "lora_target": "Target modules for LoRA ('all' or specific module names)",
            "additional_target": "Additional modules to apply LoRA",
            "lora_bf16_mode": "Whether to use bf16 for LoRA",
            "use_rslora": "Use rank stabilization for LoRA",
            "use_dora": "Use weight-decomposed LoRA (DoRA)",
        }

    def get_qlora_params(self) -> Dict[str, str]:
        """Get QLoRA-specific parameters (LoRA + quantization)."""
        params = self.get_lora_params()
        params.update(
            {
                "quantization_bit": "Number of bits for quantization (4 or 8)",
                "quantization_type": "Quantization type: 'fp4', 'nf4' (recommended), or 'int8'",
                "double_quantization": "Use double quantization for 4-bit training",
                "bnb_4bit_compute_dtype": "Compute dtype for 4-bit quantization (bf16 or fp16)",
                "bnb_4bit_quant_type": "Quantization type for 4-bit ('nf4' or 'fp4')",
                "bnb_4bit_use_double_quant": "Use nested quantization for 4-bit",
            }
        )
        return params

    def get_freeze_params(self) -> Dict[str, str]:
        """Get freeze (partial-parameter) training parameters."""
        return {
            "freeze_trainable_layers": "Number of trainable layers (positive: last n, negative: first n)",
            "freeze_trainable_modules": "Names of trainable modules (comma-separated or 'all')",
            "freeze_extra_modules": "Additional modules to train apart from main layers",
        }

    def get_full_params(self) -> Dict[str, str]:
        """Get full fine-tuning parameters."""
        return {
            "flash_attn": "Use Flash Attention 2 for efficiency",
            "disable_gradient_checkpointing": "Disable gradient checkpointing (uses more memory but faster)",
            "upcast_layernorm": "Upcast layer norm to fp32 for stability",
            "upcast_lmhead_output": "Upcast language model head output to fp32",
        }

    def get_params_for_method(self, method: str) -> Dict[str, str]:
        """Get all relevant parameters for a specific fine-tuning method.

        Args:
            method: Fine-tuning method name (e.g., 'lora', 'qlora', 'full', 'freeze')

        Returns:
            Dictionary of parameter names and their descriptions
        """
        # Start with common parameters
        all_params = self.get_common_training_params()

        # Add method-specific parameters
        method_lower = method.lower()
        if method_lower == "lora":
            all_params.update(self.get_lora_params())
        elif method_lower == "qlora":
            all_params.update(self.get_qlora_params())
        elif method_lower == "freeze":
            all_params.update(self.get_freeze_params())
        elif method_lower == "full":
            all_params.update(self.get_full_params())
        else:
            logger.warning(f"Unknown fine-tuning method: {method}")

        return all_params

    def format_params_for_prompt(self, method: str) -> str:
        """Format parameters for inclusion in LLM prompt.

        Args:
            method: Fine-tuning method name

        Returns:
            Formatted string with parameter descriptions
        """
        params = self.get_params_for_method(method)

        lines = [f"## Available Parameters for {method.upper()} Fine-tuning\n"]

        # Group parameters by category
        common_params = []
        method_specific_params = []

        common_keys = set(self.get_common_training_params().keys())

        for param, desc in sorted(params.items()):
            if param in common_keys:
                common_params.append(f"- `{param}`: {desc}")
            else:
                method_specific_params.append(f"- `{param}`: {desc}")

        if common_params:
            lines.append("### Common Training Parameters")
            lines.extend(common_params)
            lines.append("")

        if method_specific_params:
            lines.append(f"### {method.upper()}-Specific Parameters")
            lines.extend(method_specific_params)
            lines.append("")

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    query = LLaMAFactoryParamsQuery()

    # Test different methods
    for method in ["lora", "qlora", "full", "freeze"]:
        print(f"\n{'='*60}")
        print(query.format_params_for_prompt(method))
