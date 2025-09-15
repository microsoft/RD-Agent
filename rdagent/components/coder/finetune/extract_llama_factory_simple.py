"""
Simple and clean LLaMA Factory parameter extraction script.
Extracts and categorizes parameters for efficient use.
"""

import json
import sys
import time
from dataclasses import fields
from pathlib import Path

# Add LLaMA Factory to path
sys.path.insert(0, "/workspace/LLaMA-Factory/src")

from llamafactory.extras.constants import METHODS, SUPPORTED_MODELS, TRAINING_STAGES

# Import all necessary components
from llamafactory.hparams.data_args import DataArguments
from llamafactory.hparams.finetuning_args import (
    FinetuningArguments,
    FreezeArguments,
    LoraArguments,
)
from llamafactory.hparams.model_args import ModelArguments, QuantizationArguments
from transformers import TrainingArguments


def extract_field_info(field):
    """Extract field information in a simple format."""
    return {
        "name": field.name,
        "type": str(field.type).replace("typing.", "").replace("<class '", "").replace("'>", ""),
        "default": field.default if hasattr(field, "default") else None,
        "help": field.metadata.get("help", "") if field.metadata else "",
    }


def extract_params(cls):
    """Extract all parameters from a dataclass."""
    return {field.name: extract_field_info(field) for field in fields(cls)}


def save_categorized(data, base_dir):
    """Save data in categorized structure."""
    base_path = Path(base_dir)

    # Save metadata
    (base_path / "metadata.json").write_text(
        json.dumps(
            {"timestamp": int(time.time()), "version": "2.0", "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")},
            indent=2,
        )
    )

    # Save constants
    constants_dir = base_path / "constants"
    constants_dir.mkdir(exist_ok=True)

    (constants_dir / "methods.json").write_text(json.dumps(list(METHODS), indent=2))
    (constants_dir / "training_stages.json").write_text(json.dumps(dict(TRAINING_STAGES), indent=2))
    (constants_dir / "supported_models.json").write_text(
        json.dumps(dict(SUPPORTED_MODELS) if SUPPORTED_MODELS else {}, indent=2)
    )

    # Save common parameters
    common_dir = base_path / "common"
    common_dir.mkdir(exist_ok=True)

    (common_dir / "model.json").write_text(json.dumps(extract_params(ModelArguments), indent=2))
    (common_dir / "data.json").write_text(json.dumps(extract_params(DataArguments), indent=2))

    # Extract only essential training parameters
    training_params = extract_params(TrainingArguments)
    essential_keys = [
        "output_dir",
        "overwrite_output_dir",
        "do_train",
        "do_eval",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "num_train_epochs",
        "lr_scheduler_type",
        "warmup_ratio",
        "logging_steps",
        "save_steps",
        "save_total_limit",
        "seed",
        "bf16",
        "fp16",
        "report_to",
        "resume_from_checkpoint",
        "push_to_hub",
    ]
    filtered_training = {k: v for k, v in training_params.items() if k in essential_keys}
    (common_dir / "training.json").write_text(json.dumps(filtered_training, indent=2))

    # Save method-specific parameters
    method_dir = base_path / "method_specific"
    method_dir.mkdir(exist_ok=True)

    (method_dir / "lora.json").write_text(json.dumps(extract_params(LoraArguments), indent=2))
    (method_dir / "freeze.json").write_text(json.dumps(extract_params(FreezeArguments), indent=2))
    (method_dir / "quantization.json").write_text(json.dumps(extract_params(QuantizationArguments), indent=2))

    # Save SFT-specific parameters from FinetuningArguments
    stage_dir = base_path / "stage_specific"
    stage_dir.mkdir(exist_ok=True)

    finetune_params = extract_params(FinetuningArguments)
    sft_keys = ["stage", "finetuning_type", "pure_bf16", "compute_accuracy", "plot_loss"]
    sft_params = {k: v for k, v in finetune_params.items() if k in sft_keys}
    (stage_dir / "sft.json").write_text(json.dumps(sft_params, indent=2))


if __name__ == "__main__":
    base_dir = "/workspace/.llama_factory_info"
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    save_categorized({}, base_dir)
    print("SUCCESS")
