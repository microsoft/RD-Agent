#!/usr/bin/env python
"""
OpenCompass Benchmark Evaluation Entrypoint

This script runs inside the OpenCompass Docker container.
It reads configurations from environment variables and dynamically generates
a Python config file for OpenCompass.
"""

import os
import sys
from pathlib import Path


def main():
    """Generate OpenCompass config and run evaluation."""
    # Read configurations from environment variables
    datasets = os.environ.get("BENCHMARK_DATASETS", "mmlu").split(",")
    adapter_path = os.environ.get("ADAPTER_PATH", "/workspace/output")
    base_model = os.environ.get("BASE_MODEL", "")
    model_abbr = os.environ.get("MODEL_ABBR", "finetuned-model")

    # Optional parameters with defaults
    max_out_len = int(os.environ.get("MAX_OUT_LEN", "2048"))
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    num_gpus = int(os.environ.get("NUM_GPUS", "1"))

    print("=" * 60)
    print("OpenCompass Benchmark Evaluation")
    print("=" * 60)
    print(f"Model Abbreviation: {model_abbr}")
    print(f"Base Model: {base_model}")
    print(f"Adapter Path: {adapter_path}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Config: max_out_len={max_out_len}, batch_size={batch_size}, num_gpus={num_gpus}")
    print("=" * 60)

    # Generate OpenCompass Python config
    # This follows the official config format from OpenCompass
    config_lines = [
        "# Auto-generated OpenCompass config",
        "from opencompass.models import HuggingFacewithChatTemplate",
        "",
        "# Model configuration",
        "models = [dict(",
        "    type=HuggingFacewithChatTemplate,",
        f"    abbr='{model_abbr}',",  # Required: for result identification
        f"    path='{base_model}',",  # Required: base model
        f"    peft_path='{adapter_path}',",  # Required: adapter path
        f"    tokenizer_path='{adapter_path}',",  # Use adapter's tokenizer (may have new tokens)
        f"    max_out_len={max_out_len},",  # Maximum output length
        f"    batch_size={batch_size},",  # Batch size for inference
        f"    run_cfg=dict(num_gpus={num_gpus}),",  # GPU allocation
        "    model_kwargs=dict(",  # Model loading parameters
        "        device_map='auto',",  # Automatic device placement
        "        trust_remote_code=True,",  # Allow custom model code
        "    ),",
        "    generation_kwargs=dict(",  # Generation parameters
        "        do_sample=False,",  # Deterministic generation for benchmarking
        "    ),",
        ")]",
        "",
        "# Dataset configuration",
        "datasets = []",
    ]

    # Dataset mapping to OpenCompass preset configs
    # These imports load standard dataset configurations
    dataset_map = {
        "mmlu": "from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets\ndatasets.extend(mmlu_datasets)",
        "gsm8k": "from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets\ndatasets.extend(gsm8k_datasets)",
        "humaneval": "from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets\ndatasets.extend(humaneval_datasets)",
        "cmmlu": "from opencompass.configs.datasets.cmmlu.cmmlu_gen import cmmlu_datasets\ndatasets.extend(cmmlu_datasets)",
        "bbh": "from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets\ndatasets.extend(bbh_datasets)",
        "hellaswag": "from opencompass.configs.datasets.hellaswag.hellaswag_gen import hellaswag_datasets\ndatasets.extend(hellaswag_datasets)",
        "arc": "from opencompass.configs.datasets.ARC.ARC_gen import ARC_datasets\ndatasets.extend(ARC_datasets)",
    }

    # Add requested datasets
    for ds in datasets:
        ds_clean = ds.strip().lower()
        if ds_clean in dataset_map:
            config_lines.append(dataset_map[ds_clean])
        else:
            print(f"Warning: Unknown dataset '{ds_clean}', skipping")
            print(f"Supported datasets: {', '.join(dataset_map.keys())}")

    # Write config file
    config_file = Path("/workspace/benchmark_config.py")
    config_file.write_text("\n".join(config_lines))
    print(f"\nGenerated config file: {config_file}")
    print("\nConfig content:")
    print("-" * 60)
    print(config_file.read_text())
    print("-" * 60)

    # Run OpenCompass CLI
    sys.argv = [
        "opencompass",
        str(config_file),
        "--work-dir",
        "/workspace/benchmark_results",
    ]

    try:
        from opencompass.cli.main import main as opencompass_main

        print("\nStarting OpenCompass evaluation...\n")
        opencompass_main()
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR: OpenCompass evaluation failed")
        print(f"{'=' * 60}")
        print(f"{type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
