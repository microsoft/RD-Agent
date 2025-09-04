"""Utilities for fine-tuning scenario data extraction and analysis."""

import json
from pathlib import Path
from typing import Any

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.scen.utils import (
    DataFolderDescriptor,
    FileTreeGenerator,
)
from rdagent.scenarios.finetune.utils import prev_model_dirname
from rdagent.utils.agent.tpl import T


def _discover_data_files_recursive(dataset_path: Path, max_depth: int = 3) -> list[Path]:
    """
    Recursively discover data files in dataset directory.

    Args:
        dataset_path: Root path of the dataset
        max_depth: Maximum depth to search (prevents infinite recursion)

    Returns:
        List of Path objects for discovered data files, prioritized by directory level
    """
    data_patterns = ["*.json", "*.jsonl", "*.csv", "*.txt", "*.parquet"]
    found_files = []

    def search_directory(path: Path, current_depth: int = 0):
        if current_depth > max_depth:
            return

        # Search for data files in current directory
        current_files = []
        for pattern in data_patterns:
            current_files.extend(path.glob(pattern))

        # Add files from current directory (with priority info)
        for file_path in current_files:
            found_files.append((current_depth, file_path))

        # Recursively search subdirectories
        if current_depth < max_depth:
            try:
                for subdir in path.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith("."):
                        search_directory(subdir, current_depth + 1)
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot access directory {path}: {e}")

    search_directory(dataset_path)

    # Sort by depth (prioritize files in root directory) and then by name
    found_files.sort(key=lambda x: (x[0], x[1].name))

    return [file_path for _, file_path in found_files]


def extract_dataset_info(competition: str) -> dict[str, Any]:
    """Extract dataset information from files and metadata."""
    if not FT_RD_SETTING.file_path:
        return {"name": competition, "description": "FT_FILE_PATH not set", "samples": [], "files": []}

    dataset_path = Path(FT_RD_SETTING.file_path) / "datasets" / competition
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

    # Discover data files recursively
    discovered_files = _discover_data_files_recursive(dataset_path)

    # Store relative paths for cleaner display
    info["files"] = []
    for file_path in discovered_files:
        try:
            relative_path = file_path.relative_to(dataset_path)
            info["files"].append(str(relative_path))
        except ValueError:
            # Fallback to absolute path if relative path calculation fails
            info["files"].append(file_path.name)

    # Extract samples from first data file
    if discovered_files:
        _extract_data_samples(discovered_files[0], info)
        logger.info(f"Extracted samples from: {discovered_files[0].relative_to(dataset_path)}")

    return info


def extract_model_info(base_model_name: str = None) -> dict[str, Any]:
    """Extract model information from config and metadata."""
    model_name = base_model_name or FT_RD_SETTING.base_model
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


def _truncate_long_values(obj, max_length: int = 200):
    """Recursively truncate long string values in nested data structures."""
    if isinstance(obj, dict):
        return {k: _truncate_long_values(v, max_length) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_truncate_long_values(item, max_length) for item in obj]
    elif isinstance(obj, str) and len(obj) > max_length:
        return obj[:max_length] + "..."
    return obj


def _extract_data_samples(file_path: Path, info: dict[str, Any]) -> None:
    """Extract sample data from file for analysis using shared descriptor functionality."""
    try:
        descriptor = FinetuneDatasetDescriptor()
        samples_str = descriptor._extract_samples_for_prompt(file_path)
        if samples_str:
            # Parse the JSON string back to objects for info dict
            samples = json.loads(samples_str)[:2]  # First 2 samples for consistency
            info["samples"] = _truncate_long_values(samples)  # Truncate long values for prompt
    except Exception as e:
        logger.warning(f"Failed to extract samples from {file_path}: {e}")


def _find_model_path() -> Path | None:
    """Find model directory in FT_FILE_PATH structure."""
    if not FT_RD_SETTING.file_path or not FT_RD_SETTING.base_model:
        return None

    candidates = [
        Path(FT_RD_SETTING.file_path) / "models" / FT_RD_SETTING.base_model,
        Path(FT_RD_SETTING.file_path)
        / "prev_model"
        / prev_model_dirname(FT_RD_SETTING.base_model, FT_RD_SETTING.dataset),
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
    from rdagent.scenarios.data_science.scen.utils import describe_data_folder_v2

    try:
        # Use FT_FILE_PATH structure: /path/to/finetune/datasets/<dataset>
        if not FT_RD_SETTING.file_path:
            return "FT_FILE_PATH environment variable not set"

        ft_root = Path(FT_RD_SETTING.file_path)
        if not ft_root.exists():
            return f"FT_FILE_PATH does not exist: {ft_root}"

        if dataset:
            # Describe specific dataset directory
            dataset_path = Path(FT_RD_SETTING.file_path) / "datasets" / dataset
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


class FinetuneDatasetDescriptor(DataFolderDescriptor):
    """Specialized dataset descriptor for finetune scenarios that provides separated file tree and data samples."""

    def get_separated_info(self, dataset_path: Path) -> tuple[str, str]:
        """Get file tree and data samples separately for finetune prompt generation.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            tuple: (file_tree, data_samples)
        """
        try:
            # Generate file tree
            file_tree = self._get_file_tree(dataset_path)

            # Get real data samples using inherited preview functions
            data_samples = self._get_dataset_samples(dataset_path)

            return file_tree, data_samples

        except Exception as e:
            logger.warning(f"Could not generate separated dataset info: {e}")
            error_msg = f"Error: {str(e)}"
            return error_msg, error_msg

    def _get_file_tree(self, dataset_path: Path) -> str:
        """Generate file tree for the dataset directory."""
        try:
            generator = FileTreeGenerator(max_lines=150)
            return generator.generate_tree(dataset_path)

        except Exception as e:
            logger.warning(f"Could not generate file tree: {e}")
            return f"Error generating file tree: {str(e)}"

    def _get_dataset_samples(self, dataset_path: Path) -> str:
        """Extract real data samples from dataset files using recursive search and inherited preview functions."""
        try:
            # Use the intelligent recursive file discovery function
            data_files = _discover_data_files_recursive(dataset_path, max_depth=3)

            if not data_files:
                return f"No supported data files found in {dataset_path}"

            # Collect file paths for summary
            sample_file_paths = []
            samples = []

            # Process up to 3 files to get samples
            for data_file in data_files[:3]:
                try:
                    # Use inherited preview functions but extract just the content
                    file_samples = self._extract_samples_for_prompt(data_file)
                    if file_samples:
                        # Show relative path with context about input directory
                        try:
                            relative_path = data_file.relative_to(dataset_path)
                            file_label = f"Input file: {relative_path}"
                            sample_file_paths.append(str(relative_path))
                        except ValueError:
                            file_label = f"Input file: {data_file.name}"
                            sample_file_paths.append(data_file.name)

                        samples.append(f"{file_label}:\n{file_samples}")
                except Exception as e:
                    logger.warning(f"Could not extract samples from {data_file.name}: {e}")
                    continue

            if samples:
                # Add file paths summary at the beginning
                paths_summary = (
                    f"Sampled File Paths:\n"
                    + "\n".join([f"- {path}" for path in sample_file_paths])
                    + "\n\nSample Data Content:\n\n"
                )
                return paths_summary + "\n\n".join(samples)
            else:
                return "No data samples could be extracted"

        except Exception as e:
            logger.warning(f"Could not load dataset samples: {e}")
            return f"Error loading samples: {str(e)}"

    def _extract_samples_for_prompt(self, data_file: Path) -> str:
        """Extract samples formatted for LLM prompts using inherited preview functions."""
        import json

        import pandas as pd

        try:
            if data_file.suffix.lower() == ".json":
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        # Get first 3 samples and truncate long values
                        samples = _truncate_long_values(data[:3])
                        return json.dumps(samples, ensure_ascii=False, indent=2)
                    elif isinstance(data, dict):
                        truncated_data = _truncate_long_values(data)
                        return json.dumps(truncated_data, ensure_ascii=False, indent=2)

            elif data_file.suffix.lower() == ".jsonl":
                samples = []
                with open(data_file, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= 3:  # Only get first 3 samples
                            break
                        line = line.strip()
                        if line:
                            samples.append(json.loads(line))
                if samples:
                    truncated_samples = _truncate_long_values(samples)
                    return json.dumps(truncated_samples, ensure_ascii=False, indent=2)

            elif data_file.suffix.lower() == ".csv":
                df = pd.read_csv(data_file)
                if len(df) > 0:
                    samples = df.head(3).to_dict("records")
                    truncated_samples = _truncate_long_values(samples)
                    return json.dumps(truncated_samples, ensure_ascii=False, indent=2)

            elif data_file.suffix.lower() == ".parquet":
                df = pd.read_parquet(data_file)
                if len(df) > 0:
                    samples = df.head(3).to_dict("records")
                    truncated_samples = _truncate_long_values(samples)
                    return json.dumps(truncated_samples, ensure_ascii=False, indent=2)

            return ""

        except Exception as e:
            logger.warning(f"Error extracting samples from {data_file.name}: {e}")
            return ""


def generate_dataset_info_config(dataset: str, ft_file_path: str) -> dict:
    """Generate dataset_info.json configuration entry using AI for LLaMA-Factory compatibility.

    Args:
        dataset: Name of the dataset
        ft_file_path: Path to finetune directory structure

    Returns:
        dict: Configuration entry for dataset_info.json

    Raises:
        RuntimeError: If configuration generation or validation fails
    """
    from rdagent.oai.llm_utils import APIBackend

    dataset_path = Path(ft_file_path) / "datasets" / dataset

    # Use existing descriptor to get dataset information
    descriptor = FinetuneDatasetDescriptor()
    file_tree, data_samples = descriptor.get_separated_info(dataset_path)

    # Create prompt using template
    system_prompt = T("scenarios.finetune.scen.prompts:dataset_info_generation.system").r()
    # TODO: select appropriate columns (Reasoning first?)
    # TODO: guide llm: how to select dir? (not enabled yet)
    user_prompt = T("scenarios.finetune.scen.prompts:dataset_info_generation.user").r(
        dataset=dataset, file_tree=file_tree, data_samples=data_samples
    )

    # Generate configuration using API
    api = APIBackend()
    raw_response = api.build_messages_and_create_chat_completion(
        system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True
    )

    response_dict = json.loads(raw_response)

    # Extract and validate configuration
    if dataset not in response_dict:
        raise RuntimeError(f"Generated response missing key '{dataset}'")

    config = response_dict[dataset]
    if not _validate_dataset_config(config):
        raise RuntimeError(f"Invalid configuration for '{dataset}'")

    logger.info(f"Generated configuration for '{dataset}'")
    return config


def _validate_dataset_config(config: dict) -> bool:
    """Validate generated dataset configuration for LLaMA-Factory compliance.

    Args:
        config: Configuration dictionary to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(config, dict):
        logger.error("Configuration must be a dictionary")
        return False

    if "file_name" not in config:
        logger.error("Configuration must contain 'file_name' field")
        return False

    file_name = config["file_name"]
    if not isinstance(file_name, (str, list)):
        logger.error("'file_name' must be a string or list of strings")
        return False

    # Validate file_name format
    if isinstance(file_name, list):
        for fn in file_name:
            if not isinstance(fn, str) or fn.startswith("/"):
                logger.error("file_name entries must be relative paths (not absolute)")
                return False
    elif isinstance(file_name, str) and file_name.startswith("/"):
        logger.error("file_name must be a relative path (not absolute)")
        return False

    formatting = config.get("formatting", "alpaca")
    if formatting not in ["alpaca", "sharegpt"]:
        logger.error(f"Invalid formatting: {formatting}. Must be 'alpaca' or 'sharegpt'")
        return False

    if "columns" in config and not isinstance(config["columns"], dict):
        logger.error("'columns' field must be a dictionary")
        return False

    logger.info("Dataset configuration validation passed")
    return True
