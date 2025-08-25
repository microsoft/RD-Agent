"""Utilities for fine-tuning scenario data extraction and analysis."""

import json
from pathlib import Path
from typing import Any

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.utils import prev_model_dirname
from rdagent.utils.agent.tpl import T


def get_unified_mount_volumes() -> dict:
    """Get unified mount volume configuration for LLM finetune environments.

    Mount FT_FILE_PATH to /data in Docker to avoid conflict with /workspace mount.
    The /workspace is used for temporary code execution, while /data contains datasets.

    Returns:
        Dictionary of local_path -> docker_path mappings
    """
    import os

    ft_file_path = os.environ.get("FT_FILE_PATH")
    if ft_file_path and Path(ft_file_path).exists():
        return {ft_file_path: "/data"}

    # Fallback: use local_data_path if FT_FILE_PATH not available
    if FT_RD_SETTING.local_data_path and Path(FT_RD_SETTING.local_data_path).exists():
        return {FT_RD_SETTING.local_data_path: "/data"}

    return {}


def extract_dataset_info(competition: str) -> dict[str, Any]:
    """Extract dataset information from files and metadata."""
    import os

    ft_file_path = os.environ.get("FT_FILE_PATH")
    if not ft_file_path:
        return {"name": competition, "description": "FT_FILE_PATH not set", "samples": [], "files": []}

    dataset_path = Path(ft_file_path) / "dataset" / competition
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


def extract_model_info(base_model_name: str = None) -> dict[str, Any]:
    """Extract model information from config and metadata."""
    model_name = base_model_name or FT_RD_SETTING.base_model_name
    info = {
        "name": model_name or "Unknown",
        "description": "",
        "specs": "",
    }

    if not model_name:
        return info

    # Find model path
    model_path = _find_model_path()
    if not model_path:
        return info

    # Read config
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
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


def _extract_data_samples(file_path: Path, info: dict[str, Any]) -> None:
    """Extract sample data from file for analysis."""
    try:
        if file_path.suffix == ".json":
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    info["samples"] = data[:2]  # First 2 samples
        elif file_path.suffix == ".jsonl":
            with open(file_path, encoding="utf-8") as f:
                info["samples"] = [json.loads(line) for i, line in enumerate(f) if i < 2]
        elif file_path.suffix == ".csv":
            import pandas as pd

            df = pd.read_csv(file_path, nrows=2)
            info["samples"] = df.to_dict("records")
    except Exception as e:
        logger.warning(f"Failed to extract samples from {file_path}: {e}")


def _find_model_path() -> Path | None:
    """Find model directory in FT_FILE_PATH structure."""
    import os

    ft_file_path = os.environ.get("FT_FILE_PATH")
    if not ft_file_path or not FT_RD_SETTING.base_model_name:
        return None

    ft_root = Path(ft_file_path)
    candidates = [
        ft_root / "model" / FT_RD_SETTING.base_model_name,
        ft_root / "prev_model" / prev_model_dirname(FT_RD_SETTING.base_model_name, FT_RD_SETTING.dataset),
    ]

    for path in candidates:
        if path.exists():
            return path
    return None


def build_finetune_description(dataset_info: dict[str, Any], model_info: dict[str, Any]) -> str:
    """Build comprehensive fine-tuning task description using template."""
    return T(".prompts:task_description").r(
        model_name=model_info["name"],
        dataset_name=dataset_info["name"],
        dataset_description=dataset_info.get("description", ""),
        dataset_files=dataset_info.get("files", [])[:5],  # Max 5 files
        dataset_samples=dataset_info.get("samples", []),
        model_specs=model_info.get("specs", ""),
        model_description=model_info.get("description", ""),
    )


def build_folder_description(dataset: str = None) -> str:
    """Generate folder description using describe_data_folder_v2, consistent with data science scenario."""
    import os

    from rdagent.scenarios.data_science.scen.utils import describe_data_folder_v2

    try:
        # Use FT_FILE_PATH structure: /path/to/finetune/dataset/<dataset>
        ft_file_path = os.environ.get("FT_FILE_PATH")
        if not ft_file_path:
            return "FT_FILE_PATH environment variable not set"

        ft_root = Path(ft_file_path)
        if not ft_root.exists():
            return f"FT_FILE_PATH does not exist: {ft_root}"

        if dataset:
            # Describe specific dataset directory
            dataset_path = ft_root / "dataset" / dataset
        else:
            # Describe entire finetune directory structure
            dataset_path = ft_root

        if not dataset_path.exists():
            return f"Dataset path {dataset_path} does not exist"

        # Directly call describe_data_folder_v2, same as data science scenario
        return describe_data_folder_v2(dataset_path, show_nan_columns=FT_RD_SETTING.show_nan_columns, max_length=20000)

    except Exception as e:
        logger.warning(f"Failed to generate folder description: {e}")
        return f"Error generating folder description: {str(e)}"
