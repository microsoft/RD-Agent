"""
Streamlined LLaMA Factory parameter extraction script.
Extracts all parameters directly from LLaMA Factory without hardcoded filtering.
Always pulls the latest LLaMA Factory code before extraction.
"""

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from llamafactory.extras.constants import METHODS, SUPPORTED_MODELS, TRAINING_STAGES
from llamafactory.data.template import TEMPLATES
from llamafactory.hparams.data_args import DataArguments
from llamafactory.hparams.finetuning_args import FinetuningArguments, FreezeArguments, LoraArguments
from llamafactory.hparams.model_args import ModelArguments, QuantizationArguments
from transformers import TrainingArguments


# Pull latest LLaMA Factory code
try:
    result = subprocess.run(
        ["git", "pull", "--rebase"],
        cwd="/llamafactory",
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        output = (result.stdout or result.stderr).strip()
        print(f"Updated LLaMA Factory: {output}", file=sys.stderr)
    else:
        print(f"Warning: git pull failed: {result.stderr}", file=sys.stderr)
except Exception as e:
    print(f"Warning: Failed to update LLaMA Factory: {e}", file=sys.stderr)

# Add LLaMA Factory to path
sys.path.insert(0, "/llamafactory/src")


def extract_field_info(field):
    """Extract field information from a dataclass field."""
    from dataclasses import MISSING
    
    # Handle default value - avoid MISSING type which is not JSON serializable
    if hasattr(field, "default") and field.default is not MISSING:
        default_value = field.default
    elif hasattr(field, "default_factory") and field.default_factory is not MISSING:
        default_value = "<factory>"
    else:
        default_value = None
    
    return {
        "name": field.name,
        "type": str(field.type).replace("typing.", "").replace("<class '", "").replace("'>", ""),
        "default": default_value,
        "help": field.metadata.get("help", "") if field.metadata else "",
    }


def extract_params(cls):
    """Extract all parameters from a dataclass."""
    return {field.name: extract_field_info(field) for field in fields(cls)}


def save_parameters(base_dir):
    """Extract and save all LLaMA Factory parameters in a flat structure."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Save constants
    constants = {
        "methods": list(METHODS),
        "training_stages": dict(TRAINING_STAGES),
        "supported_models": dict(SUPPORTED_MODELS) if SUPPORTED_MODELS else {},
        "templates": list(TEMPLATES.keys()),
    }
    (base_path / "constants.json").write_text(json.dumps(constants, indent=2))

    # Save parameters - extract ALL parameters without filtering
    parameters = {
        "model": extract_params(ModelArguments),
        "data": extract_params(DataArguments),
        "training": extract_params(TrainingArguments),
        "finetuning": {
            **extract_params(FinetuningArguments),
            **extract_params(LoraArguments),
            **extract_params(FreezeArguments),
            **extract_params(QuantizationArguments),
        },
    }
    (base_path / "parameters.json").write_text(json.dumps(parameters, indent=2))


def main():
    """Main entry point for parameter extraction."""
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/.llama_factory_info"
    
    try:
        save_parameters(base_dir)
        print("SUCCESS")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
