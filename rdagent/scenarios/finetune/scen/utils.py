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

    This function provides a centralized way to configure volume mounting
    for consistency across all LLM finetune components.

    Returns:
        Dictionary of local_path -> docker_path mappings
    """
    extra_volumes = {}

    if FT_RD_SETTING.file_path:
        # Mount entire dataset and model directories
        dataset_root = f"{FT_RD_SETTING.file_path}/dataset"
        model_root = f"{FT_RD_SETTING.file_path}/model"

        if Path(dataset_root).exists():
            extra_volumes[dataset_root] = "/workspace/dataset"
        if Path(model_root).exists():
            extra_volumes[model_root] = "/workspace/model"

    # Fallback: use local_data_path if available
    if FT_RD_SETTING.local_data_path and Path(FT_RD_SETTING.local_data_path).exists():
        extra_volumes[FT_RD_SETTING.local_data_path] = "/workspace/dataset"

    return extra_volumes


def extract_dataset_info(competition: str) -> dict[str, Any]:
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
    """Generate folder description by running describe_data_folder_v2 inside Docker environment."""
    from rdagent.components.coder.finetune.conf import get_ft_env

    # Use unified mount volume configuration
    extra_volumes = get_unified_mount_volumes()

    # Create temporary environment for folder description
    env = get_ft_env(
        extra_volumes=extra_volumes,
        running_timeout_period=300,  # 5 minutes should be enough for folder description
        enable_cache=False,
    )

    try:
        # Use simplified Docker path structure
        docker_path = f"/workspace/dataset/{dataset}" if dataset else "/workspace/dataset"

        # Execute folder description generation inside Docker
        cmd = (
            f'python -c "'
            f"from rdagent.scenarios.data_science.scen.utils import describe_data_folder_v2; "
            f"print(describe_data_folder_v2("
            f'\\"{docker_path}\\", '
            f"show_nan_columns={FT_RD_SETTING.show_nan_columns}, "
            f'max_length=20000))"'
        )

        result = env.run(cmd)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            # Fallback: try to describe the root mounted directory
            fallback_cmd = (
                f'python -c "'
                f"from rdagent.scenarios.data_science.scen.utils import describe_data_folder_v2; "
                f"print(describe_data_folder_v2("
                f'\\"/workspace/dataset\\", '
                f"show_nan_columns={FT_RD_SETTING.show_nan_columns}, "
                f'max_length=20000))"'
            )
            fallback_result = env.run(fallback_cmd)
            if fallback_result.returncode == 0:
                return fallback_result.stdout.strip()
            else:
                return f"Error generating folder description: {result.stderr}"

    except Exception as e:
        return f"Error running folder description in Docker: {str(e)}"
    finally:
        # Clean up environment
        try:
            env.close()
        except:
            pass
