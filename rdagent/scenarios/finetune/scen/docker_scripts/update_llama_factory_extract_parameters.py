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

import requests
from llamafactory.data.template import TEMPLATES
from llamafactory.extras.constants import METHODS, SUPPORTED_MODELS, TRAINING_STAGES
from llamafactory.hparams.data_args import DataArguments
from llamafactory.hparams.finetuning_args import (
    ApolloArguments,
    BAdamArgument,
    FinetuningArguments,
    FreezeArguments,
    GaloreArguments,
    LoraArguments,
    OFTArguments,
    RLHFArguments,
    SwanLabArguments,
)
from llamafactory.hparams.model_args import ModelArguments, QuantizationArguments
from transformers import TrainingArguments


def update_llama_factory():
    """Pull the latest LLaMA Factory code from GitHub."""
    response = requests.get(f"https://api.github.com/repos/hiyouga/LLaMA-Factory/releases/latest", timeout=10)
    latest_tag = response.json()["tag_name"]
    # Fetch tags and checkout latest release, printing both stdout and stderr.
    for cmd in (["git", "fetch", "--tags"], ["git", "checkout", latest_tag]):
        print(f"Running: {' '.join(cmd)}", flush=True)
        result = subprocess.run(
            cmd,
            cwd="/llamafactory",
            text=True,
            timeout=120,
            capture_output=True,
            check=False,
        )
        print(f"Exit code: {result.returncode}", flush=True)
        print("STDOUT:", result.stdout.strip() or "<empty>", flush=True)
        print("STDERR:", result.stderr.strip() or "<empty>", flush=True)
        if result.returncode != 0:
            print("Command failed, continuing.", flush=True)

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


def extract_base_params(cls):
    """Extract only the parameters defined in the class itself, not inherited."""
    # Get all fields from the class
    all_fields = {f.name: f for f in fields(cls)}

    # Get fields from all parent classes
    parent_fields = set()
    for base in cls.__bases__:
        if hasattr(base, "__dataclass_fields__"):
            parent_fields.update(base.__dataclass_fields__.keys())

    # Keep only fields defined in the class itself
    own_fields = {name: field for name, field in all_fields.items() if name not in parent_fields}

    return {name: extract_field_info(field) for name, field in own_fields.items()}


def save_parameters(base_dir):
    """Extract and save all LLaMA Factory parameters with category information."""
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

    # Save parameters - preserve parameter ownership by categorizing them
    parameters = {
        "model": extract_params(ModelArguments),
        "data": extract_params(DataArguments),
        "training": extract_params(TrainingArguments),
        "finetuning": {
            # Categorize parameters by PEFT method
            "freeze": extract_params(FreezeArguments),
            "lora": extract_params(LoraArguments),
            "oft": extract_params(OFTArguments),
            "galore": extract_params(GaloreArguments),
            "apollo": extract_params(ApolloArguments),
            "badam": extract_params(BAdamArgument),
            "rlhf": extract_params(RLHFArguments),
            "swanlab": extract_params(SwanLabArguments),
            "quantization": extract_params(QuantizationArguments),
            # Extract only FinetuningArguments' own parameters (excluding inherited ones)
            "base": extract_base_params(FinetuningArguments),
        },
    }
    (base_path / "parameters.json").write_text(json.dumps(parameters, indent=2))


def main():
    """Main entry point for parameter extraction."""
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/.llama_factory_info"

    try:
        update_llama_factory()
        save_parameters(base_dir)
        print("Successfully extracted LLaMA Factory parameters")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
