#!/usr/bin/env python
"""OpenCompass Benchmark Evaluation Entrypoint."""

import json
import os
import sys
from pathlib import Path


def main():
    """Run OpenCompass evaluation."""
    # Read config from environment or file
    datasets = os.environ.get("BENCHMARK_DATASETS", "mmlu").split(",")
    adapter_path = os.environ.get("ADAPTER_PATH", "/workspace/output")
    base_model = os.environ.get("BASE_MODEL", "")

    print(f"Benchmark Evaluation")
    print(f"Datasets: {datasets}")
    print(f"Adapter: {adapter_path}")
    print(f"Base Model: {base_model}")

    # Generate OpenCompass config
    config_lines = [
        "from opencompass.models import HuggingFacewithChatTemplate",
        "",
        "models = [dict(",
        "    type=HuggingFacewithChatTemplate,",
        f"    path='{base_model}',",
        f"    peft_path='{adapter_path}',",
        "    max_out_len=2048,",
        "    batch_size=8,",
        ")]",
        "",
        "datasets = []",
    ]

    # Dataset mapping (simplified)
    dataset_map = {
        "mmlu": "from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets\ndatasets.extend(mmlu_datasets)",
        "gsm8k": "from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets\ndatasets.extend(gsm8k_datasets)",
        "cmmlu": "from opencompass.configs.datasets.cmmlu.cmmlu_gen import cmmlu_datasets\ndatasets.extend(cmmlu_datasets)",
        "humaneval": "from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets\ndatasets.extend(humaneval_datasets)",
        "bbh": "from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets\ndatasets.extend(bbh_datasets)",
    }

    for ds in datasets:
        if ds in dataset_map:
            config_lines.append(dataset_map[ds])

    # Write config
    config_file = Path("/workspace/benchmark_config.py")
    config_file.write_text("\n".join(config_lines))

    # Run OpenCompass
    sys.argv = ["opencompass", str(config_file), "--work-dir", "/workspace/benchmark_results"]

    try:
        from opencompass.cli.main import main as opencompass_main

        opencompass_main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
