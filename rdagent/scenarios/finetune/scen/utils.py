"""Utilities for fine-tuning scenario data extraction and analysis."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.utils import prev_model_dirname


def extract_dataset_info(competition: str) -> Dict[str, Any]:
    """Extract dataset information from files and metadata."""
    dataset_path = Path(FT_RD_SETTING.local_data_path) / competition
    info = {"name": competition, "description": "", "samples": [], "files": []}

    # Read description from README
    for readme in ["README.md", "readme.md", "README.txt"]:
        readme_path = dataset_path / readme
        if readme_path.exists():
            try:
                info["description"] = readme_path.read_text(encoding="utf-8")[:1000]
                logger.info(f"Loaded dataset description from {readme}")
                break
            except Exception as e:
                logger.warning(f"Failed to read {readme}: {e}")

    # Discover data files
    for pattern in ["*.json", "*.jsonl", "*.csv", "*.txt", "*.parquet"]:
        info["files"].extend([f.name for f in dataset_path.glob(pattern)])

    # Extract samples from first data file
    if info["files"]:
        _extract_data_samples(dataset_path / info["files"][0], info)

    return info


def extract_model_info() -> Dict[str, Any]:
    """Extract model information from config and metadata."""
    info = {
        "name": FT_RD_SETTING.base_model_name or "Unknown",
        "description": "",
        "specs": "",
    }

    if not FT_RD_SETTING.base_model_name:
        return info

    # Find model path
    model_path = _find_model_path()
    if not model_path:
        return info

    # Read config
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                specs = []
                for key in [
                    "model_type",
                    "hidden_size",
                    "num_hidden_layers",
                    "vocab_size",
                ]:
                    if key in config:
                        specs.append(f"{key}: {config[key]}")
                info["specs"] = ", ".join(specs)
        except Exception as e:
            logger.warning(f"Failed to read model config: {e}")

    # Read description
    for readme in ["README.md", "readme.md", "model_card.md"]:
        readme_path = model_path / readme
        if readme_path.exists():
            try:
                info["description"] = readme_path.read_text(encoding="utf-8")[:1000]
                logger.info(f"Loaded model description from {readme}")
                break
            except Exception as e:
                logger.warning(f"Failed to read {readme}: {e}")

    return info


def _extract_data_samples(file_path: Path, info: Dict[str, Any]) -> None:
    """Extract sample data from file for analysis."""
    try:
        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    info["samples"] = data[:2]  # First 2 samples
        elif file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                info["samples"] = [json.loads(line) for i, line in enumerate(f) if i < 2]
        elif file_path.suffix == ".csv":
            import pandas as pd

            df = pd.read_csv(file_path, nrows=2)
            info["samples"] = df.to_dict("records")
    except Exception as e:
        logger.warning(f"Failed to extract samples from {file_path}: {e}")


def _find_model_path() -> Optional[Path]:
    """Find model directory in standard locations."""
    if not FT_RD_SETTING.file_path or not FT_RD_SETTING.base_model_name:
        return None

    ft_root = Path(FT_RD_SETTING.file_path)
    candidates = [
        ft_root / "model" / FT_RD_SETTING.base_model_name,
        ft_root / "prev_model" / prev_model_dirname(FT_RD_SETTING.base_model_name, FT_RD_SETTING.dataset),
    ]

    for path in candidates:
        if path.exists():
            return path
    return None


def build_finetune_description(dataset_info: Dict[str, Any], model_info: Dict[str, Any]) -> str:
    """Build comprehensive fine-tuning task description."""
    parts = [
        "# LLM Fine-tuning Task",
        "",
        f"Fine-tune model `{model_info['name']}` using dataset `{dataset_info['name']}`.",
        "",
        "## Dataset Information",
    ]

    if dataset_info["description"]:
        parts.extend(["", dataset_info["description"]])

    if dataset_info["files"]:
        parts.extend(["", f"**Data files:** {', '.join(dataset_info['files'][:5])}"])

    if dataset_info["samples"]:
        parts.extend(["", "**Sample data:**"])
        for i, sample in enumerate(dataset_info["samples"], 1):
            parts.extend([f"```json", json.dumps(sample, ensure_ascii=False, indent=2), "```"])

    parts.extend(["", "## Model Information", f"**Model:** {model_info['name']}"])

    if model_info["specs"]:
        parts.append(f"**Specifications:** {model_info['specs']}")

    if model_info["description"]:
        parts.extend(["", model_info["description"]])

    parts.extend(
        [
            "",
            "## Fine-tuning Objective",
            "Adapt the base model to perform better on the specific task defined by the dataset.",
            "The fine-tuning process should properly load the model, process the data, and optimize performance.",
        ]
    )

    return "\n".join(parts)


def build_folder_description(competition: str) -> str:
    """Build concise folder structure description for fine-tuning."""
    from rdagent.scenarios.data_science.scen.utils import describe_data_folder_v2

    dataset_path = Path(FT_RD_SETTING.local_data_path) / competition

    # Get basic folder structure
    basic_desc = describe_data_folder_v2(dataset_path, show_nan_columns=FT_RD_SETTING.show_nan_columns)

    parts = [
        "# Fine-tuning Data Structure",
        "",
        "## Dataset Files",
        basic_desc,
        "",
        "## Model Access Paths",
        "- Load training data from: `./workspace_input/`",
        "- Load pre-trained model from: `./workspace_input/prev_model/`",
        "- Alternative model path: `./workspace_input/model/`",
        "",
        "## Expected Output",
        "Save fine-tuned model to the designated output directory after training.",
    ]

    return "\n".join(parts)
