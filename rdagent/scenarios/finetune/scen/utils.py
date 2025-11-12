"""Utilities for fine-tuning scenario data extraction and analysis."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.scen.utils import FileTreeGenerator
from rdagent.utils.agent.tpl import T


def _find_data_files(dataset_path: Path, max_files: int = 50) -> list[Path]:
    """Find data files in dataset directory using recursive glob.

    Args:
        dataset_path: Root path of the dataset
        max_files: Maximum number of files to return

    Returns:
        List of Path objects for discovered data files, sorted by name
    """
    patterns = ["*.json", "*.jsonl", "*.csv", "*.txt", "*.parquet"]
    files = []
    for pattern in patterns:
        files.extend(dataset_path.rglob(pattern))
    # Sort by name for deterministic order, limit count to avoid excessive files
    return sorted(files, key=lambda x: x.name)[:max_files]


def _truncate_long_values(obj, max_length: int = 200):
    """Recursively truncate long string values in nested data structures."""
    if isinstance(obj, dict):
        return {k: _truncate_long_values(v, max_length) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_truncate_long_values(item, max_length) for item in obj]
    elif isinstance(obj, str) and len(obj) > max_length:
        return obj[:max_length] + "..."
    return obj


class FinetuneDatasetDescription(dict):
    """Specialized dataset description for finetune scenarios."""

    def __str__(self) -> str:
        """Generate human-readable description for LLM prompts."""
        parts = []

        if "file_tree" in self:
            parts.append(f"## File Tree:\n{self['file_tree']}")

        if "data_samples" in self:
            parts.append(f"## Data Samples:\n{self['data_samples']}")

        if "stats" in self:
            stats = self["stats"]
            parts.append(
                f"## Statistics:\n"
                f"- Files: {stats.get('file_count', 0)}\n"
                f"- Samples: {stats.get('sample_count', 0)}\n"
                f"- Size: {stats.get('total_size_mb', 0)} MB"
            )

        return "\n\n".join(parts) if parts else "Empty dataset description"


class FinetuneFileDescription(dict):
    """Specialized file description for finetune scenarios."""

    def __str__(self) -> str:
        """Generate human-readable file description."""
        if "samples" in self:
            return f"File: {self.get('name', 'unknown')}\n{self['samples']}"
        return f"File: {self.get('name', 'unknown')}"


class FinetuneDatasetDescriptor:
    """Specialized dataset descriptor for finetune scenarios that provides separated file tree and data samples."""

    def _format_samples_output(
        self, file_descriptions: list[tuple[Path, FinetuneFileDescription]], dataset_path: Path
    ) -> str:
        """Format multiple file descriptions into a human-readable string.

        Args:
            file_descriptions: List of (file_path, FinetuneFileDescription) tuples
            dataset_path: Base dataset path for computing relative paths

        Returns:
            Formatted string with file paths summary and sample content
        """
        if not file_descriptions:
            return "No data samples could be extracted"

        sample_file_paths = []
        sample_contents = []

        for file_path, file_desc in file_descriptions:
            samples = file_desc.get("samples")
            if not samples:
                continue

            # Get relative path
            try:
                relative_path = file_path.relative_to(dataset_path)
                path_str = str(relative_path)
            except ValueError:
                path_str = file_path.name

            sample_file_paths.append(path_str)

            # Format samples as JSON string
            samples_json = json.dumps(samples, ensure_ascii=False, indent=2)
            sample_contents.append(f"Input file: {path_str}:\n{samples_json}")

        if not sample_contents:
            return "No data samples could be extracted"

        # Build final output
        paths_summary = "Sampled File Paths:\n" + "\n".join([f"- {path}" for path in sample_file_paths])
        return paths_summary + "\n\nSample Data Content:\n\n" + "\n\n".join(sample_contents)

    def _generate_file_tree(self, dataset_path: Path) -> str:
        """Generate file tree for the dataset directory."""
        try:
            generator = FileTreeGenerator(max_lines=150)
            return generator.generate_tree(dataset_path)
        except Exception as e:
            logger.warning(f"Could not generate file tree: {e}")
            return f"Error generating file tree: {str(e)}"

    def _count_samples_in_file(self, data_file: Path) -> int:
        """Count total samples in a single data file.

        Args:
            data_file: Path to data file

        Returns:
            Total number of samples in file (0 if error or unsupported format)
        """
        suffix = data_file.suffix.lower()

        try:
            if suffix == ".json":
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return len(data)
                    elif isinstance(data, dict):
                        return 1  # Single object

            elif suffix == ".jsonl":
                with open(data_file, "r", encoding="utf-8") as f:
                    return sum(1 for line in f if line.strip())

            elif suffix in [".csv", ".parquet"]:
                df = pd.read_csv(data_file) if suffix == ".csv" else pd.read_parquet(data_file)
                return len(df)

        except Exception as e:
            logger.warning(f"Cannot count samples in {data_file.name}: {e}")

        return 0

    def _generate_stats(self, dataset_path: Path) -> dict[str, Any]:
        """Calculate dataset statistics: sample count and total file size."""
        try:
            data_files = _find_data_files(dataset_path, max_files=50)

            total_samples = 0
            total_size_bytes = 0
            file_count = len(data_files)

            for data_file in data_files:
                # Calculate file size
                try:
                    total_size_bytes += data_file.stat().st_size
                except (OSError, FileNotFoundError):
                    logger.warning(f"Cannot get size of {data_file}")

                # Count samples using unified method
                total_samples += self._count_samples_in_file(data_file)

            return {
                "sample_count": total_samples,
                "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
                "file_count": file_count,
            }

        except Exception as e:
            logger.warning(f"Failed to calculate dataset stats: {e}")
            return {
                "sample_count": 0,
                "total_size_mb": 0,
                "file_count": 0,
            }

    def describe_dataset_folder(self, dataset_path: Path, dataset_name: str = None) -> FinetuneDatasetDescription:
        """Generate complete dataset folder description.

        Args:
            dataset_path: Path to the dataset directory
            dataset_name: Name of the dataset (defaults to directory name)

        Returns:
            FinetuneDatasetDescription with comprehensive dataset information
        """
        try:
            # Generate file tree and stats
            file_tree = self._generate_file_tree(dataset_path)
            stats = self._generate_stats(dataset_path)

            # Get data files
            data_files = _find_data_files(dataset_path, max_files=50)

            # Use public interface to describe files
            file_descriptions = []
            for data_file in data_files[:3]:  # Process first 3 files for samples
                try:
                    file_desc = self.describe_data_file(data_file)
                    if file_desc.get("samples"):
                        file_descriptions.append((data_file, file_desc))
                except Exception as e:
                    logger.warning(f"Could not describe file {data_file.name}: {e}")

            # Format samples output
            data_samples_str = self._format_samples_output(file_descriptions, dataset_path)

            # Read description from README
            description = self._read_dataset_readme(dataset_path)

            # Get file list
            files = []
            for file_path in data_files:
                try:
                    relative_path = file_path.relative_to(dataset_path)
                    files.append(str(relative_path))
                except ValueError:
                    files.append(file_path.name)

            # Extract samples for template (first 2 samples from first file)
            samples = self._extract_samples_for_template(data_files, max_samples=2)

            return FinetuneDatasetDescription(
                {
                    # For new interface (generate_dataset_info_config)
                    "file_tree": file_tree,
                    "data_samples": data_samples_str,
                    "stats": stats,
                    # For templates (scenario_description, task_description)
                    "name": dataset_name or dataset_path.name,
                    "description": description,
                    "files": files,
                    "samples": samples,
                    "sample_count": stats.get("sample_count", 0),
                    "total_size_mb": stats.get("total_size_mb", 0),
                    "file_count": stats.get("file_count", 0),
                }
            )
        except Exception as e:
            logger.warning(f"Could not generate dataset folder description: {e}")
            return FinetuneDatasetDescription(
                {
                    "file_tree": f"Error: {str(e)}",
                    "data_samples": f"Error: {str(e)}",
                    "stats": {"sample_count": 0, "total_size_mb": 0, "file_count": 0},
                    "name": dataset_name or "unknown",
                    "description": "",
                    "files": [],
                    "samples": [],
                    "sample_count": 0,
                    "total_size_mb": 0,
                    "file_count": 0,
                }
            )

    def get_dataset_stats(self, dataset_path: Path) -> dict[str, Any]:
        """Calculate dataset statistics (public interface for compatibility)."""
        return self._generate_stats(dataset_path)

    def _read_dataset_readme(self, dataset_path: Path) -> str:
        """Read README description from dataset directory.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            README content (truncated to 1000 chars) or empty string
        """
        for readme in ["README.md", "readme.md", "README.txt"]:
            readme_path = dataset_path / readme
            if readme_path.exists():
                try:
                    description = readme_path.read_text(encoding="utf-8")[:1000]
                    logger.info(f"Loaded dataset description from {readme}")
                    return description
                except Exception as e:
                    logger.warning(f"Failed to read {readme}: {e}")
        return ""

    def _extract_samples_for_template(self, data_files: list[Path], max_samples: int = 2) -> list:
        """Extract samples from first data file for template usage.

        Args:
            data_files: List of data file paths
            max_samples: Maximum samples to extract

        Returns:
            List of sample dicts (may be empty if extraction fails)
        """
        if not data_files:
            return []

        try:
            first_file = data_files[0]
            suffix = first_file.suffix.lower()

            # Dispatch to appropriate handler
            if suffix == ".json":
                file_desc = self.describe_file_json(first_file, max_samples=max_samples)
            elif suffix == ".jsonl":
                file_desc = self.describe_file_jsonl(first_file, max_samples=max_samples)
            elif suffix == ".csv":
                file_desc = self.describe_file_csv(first_file, max_samples=max_samples)
            elif suffix == ".parquet":
                file_desc = self.describe_file_parquet(first_file, max_samples=max_samples)
            else:
                return []

            return file_desc.get("samples", [])

        except Exception as e:
            logger.warning(f"Failed to extract samples for template: {e}")
            return []

    def describe_model(self, base_model_name: str = None, ft_file_path: str = None) -> dict[str, Any]:
        """Extract model information from config and metadata.

        Args:
            base_model_name: Name of the base model
            ft_file_path: Path to finetune directory structure

        Returns:
            dict with model information (name, description, specs)
        """
        model_name = base_model_name or FT_RD_SETTING.base_model
        info = {
            "name": model_name or "Unknown",
            "description": "",
            "specs": "",
        }

        if not model_name:
            return info

        # Find model path
        if not ft_file_path:
            ft_file_path = FT_RD_SETTING.file_path

        if not ft_file_path:
            return info

        model_path = Path(ft_file_path) / "models" / model_name
        if not model_path.exists():
            return info

        # Read config
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
                    specs = []
                    for key in ["model_type", "hidden_size", "num_hidden_layers", "vocab_size"]:
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

    def describe_file_json(self, data_file: Path, max_samples: int = 3) -> FinetuneFileDescription:
        samples = []
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    samples = _truncate_long_values(data[:max_samples])
                elif isinstance(data, dict):
                    truncated_data = _truncate_long_values(data)
                    samples = [truncated_data]
        except Exception as e:
            logger.warning(f"Error extracting samples from {data_file.name}: {e}")

        return FinetuneFileDescription({"name": data_file.name, "type": "json", "samples": samples})

    def describe_file_jsonl(self, data_file: Path, max_samples: int = 3) -> FinetuneFileDescription:
        samples = []
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            if samples:
                samples = _truncate_long_values(samples)
        except Exception as e:
            logger.warning(f"Error extracting samples from {data_file.name}: {e}")

        return FinetuneFileDescription({"name": data_file.name, "type": "jsonl", "samples": samples})

    def describe_file_csv(self, data_file: Path, max_samples: int = 3) -> FinetuneFileDescription:
        samples = []
        try:
            df = pd.read_csv(data_file)
            if len(df) > 0:
                samples = df.head(max_samples).to_dict("records")
                samples = _truncate_long_values(samples)
        except Exception as e:
            logger.warning(f"Error extracting samples from {data_file.name}: {e}")

        return FinetuneFileDescription({"name": data_file.name, "type": "csv", "samples": samples})

    def describe_file_parquet(self, data_file: Path, max_samples: int = 3) -> FinetuneFileDescription:
        samples = []
        try:
            df = pd.read_parquet(data_file)
            if len(df) > 0:
                samples = df.head(max_samples).to_dict("records")
                samples = _truncate_long_values(samples)
        except Exception as e:
            logger.warning(f"Error extracting samples from {data_file.name}: {e}")

        return FinetuneFileDescription({"name": data_file.name, "type": "parquet", "samples": samples})

    def describe_data_file(self, data_file: Path) -> FinetuneFileDescription:
        """Describe data file based on suffix, dispatching to specific format handlers.

        This is the main public interface for describing individual data files.
        It automatically detects file type and calls the appropriate handler.

        Args:
            data_file: Path to the data file

        Returns:
            FinetuneFileDescription with file metadata and samples
        """
        suffix = data_file.suffix.lower()
        describe_map = {
            ".json": self.describe_file_json,
            ".jsonl": self.describe_file_jsonl,
            ".csv": self.describe_file_csv,
            ".parquet": self.describe_file_parquet,
        }
        describe_func = describe_map.get(suffix)
        if describe_func:
            return describe_func(data_file)
        # For unsupported file types, return basic info
        return FinetuneFileDescription({"name": data_file.name, "type": "unknown", "samples": []})


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
    # Use existing descriptor to get dataset information
    dataset_folder_desc = FinetuneDatasetDescriptor().describe_dataset_folder(Path(ft_file_path) / "datasets" / dataset)

    # Create prompt using template
    system_prompt = T("scenarios.finetune.scen.prompts:dataset_info_generation.system").r()
    # TODO: select appropriate columns (Reasoning first?)
    # TODO: guide llm: how to select dir? (not enabled yet)
    user_prompt = T("scenarios.finetune.scen.prompts:dataset_info_generation.user").r(
        dataset=dataset,
        file_tree=dataset_folder_desc.get("file_tree"),
        data_samples=dataset_folder_desc.get("data_samples"),
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
