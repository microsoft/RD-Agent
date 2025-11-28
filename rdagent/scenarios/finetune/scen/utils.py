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
    dataset_files = sorted(files, key=lambda x: x.name)[:max_files]
    return [f for f in dataset_files if f != dataset_path / "dataset_info.json"]


def _truncate_long_values(obj, max_length: int = 3000):
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

        if "file_path_to_descriptions" in self:
            for file_path, file_desc in self["file_path_to_descriptions"]:
                parts.append(f"### File path: {file_path}\n{file_desc}")

        if "readme_file_descs" in self and self["readme_file_descs"] is not None:
            parts.append(f"## Dataset readme Description:\n{self['readme_file_descs']}")

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
        output_str = f"File name: {self.get('name', 'unknown')}\nFile Type: {self.get('type', 'unknown')}"
        if "samples" in self:
            output_str += f"\nFile Samples:\n{self['samples']}"
        for k in self:
            if k not in ["name", "type", "samples"]:
                output_str += f"\n{k.capitalize()}: {self[k]}"
        return output_str


class FinetuneDatasetDescriptor:
    """Specialized dataset descriptor for finetune scenarios that provides separated file tree and data samples."""

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

    def describe_dataset_folder(
        self, dataset_path: Path, dataset_name: str | None = None, include_dataset_readme: bool = False
    ) -> FinetuneDatasetDescription:
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
            file_path_to_descriptions = []
            for data_file in data_files[: FT_RD_SETTING.data_sample_count]:  # Process first N files for samples
                try:
                    file_path_to_descriptions.append(
                        (data_file.relative_to(dataset_path), self.describe_data_file(data_file))
                    )
                except Exception as e:
                    logger.warning(f"Could not describe file {data_file.name}: {e}")

            # Read description from README
            if include_dataset_readme:
                readme_file_descs = self._read_dataset_readme(dataset_path)
            else:
                readme_file_descs = None

            # Get file list
            files = []
            for file_path in data_files:
                try:
                    relative_path = file_path.relative_to(dataset_path)
                    files.append(str(relative_path))
                except ValueError:
                    files.append(file_path.name)

            return FinetuneDatasetDescription(
                {
                    # For new interface (generate_dataset_info_config)
                    "file_tree": file_tree,
                    "file_path_to_descriptions": file_path_to_descriptions,
                    "stats": stats,
                    # For templates (scenario_description, task_description)
                    "name": dataset_name or dataset_path.name,
                    "readme_file_descs": readme_file_descs,
                    "files": files,
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
                    "readme_file_descs": None,
                    "files": [],
                    "sample_count": 0,
                    "total_size_mb": 0,
                    "file_count": 0,
                }
            )

    def get_dataset_stats(self, dataset_path: Path) -> dict[str, Any]:
        """Calculate dataset statistics (public interface for compatibility)."""
        return self._generate_stats(dataset_path)

    def _walk(self, dir_path: Path, depth: int, max_depth: int, target_names: set[str]) -> None:
        results = []
        if depth > max_depth:
            return results
        for entry in dir_path.iterdir():
            if entry.is_file():
                # 区分大小写匹配（与题目保持一致）
                if entry.name in target_names:
                    results.append(entry)
                # 如果希望大小写不敏感，可用：
                # if entry.name.lower() in {"readme.md", "readme.txt"}:
                #     results.append(entry)
            elif entry.is_dir():
                results.extend(self._walk(entry, depth + 1, max_depth, target_names))
        return results

    def _read_dataset_readme(self, dataset_path: Path, max_chars: int = 5000) -> str:
        """Read README description from dataset directory.

        Args:
            dataset_path: Path to dataset directory
            max_chars: Maximum characters to read from each README file

        Returns:
            README content (truncated to max_chars) or empty string
        """
        target_names = {"README.md", "readme.md", "README.txt"}
        readme_files = self._walk(dataset_path, depth=0, max_depth=2, target_names=target_names)
        readme_file_descs = ""
        for readme_file in readme_files:
            try:
                description = readme_file.read_text(encoding="utf-8")[:max_chars]
                logger.info(f"Loaded dataset description from {readme_file.relative_to(dataset_path)}")
                readme_file_descs += f"### From readme file: {readme_file.relative_to(dataset_path)}:\n<start_of_readme>\n{description}<end_of_readme>\n\n"
            except Exception as e:
                logger.warning(f"Failed to read {readme_file.relative_to(dataset_path)}: {e}")
        return readme_file_descs

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
                    for key in ["model_type", "max_position_embeddings"]:
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
        jsonl_shape = None
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
            jsonl_shape = (i + 1,)

        except Exception as e:
            logger.warning(f"Error extracting samples from {data_file.name}: {e}")

        return FinetuneFileDescription(
            {"name": data_file.name, "type": "jsonl", "samples": samples, "shape": jsonl_shape}
        )

    def describe_file_csv(self, data_file: Path, max_samples: int = 3) -> FinetuneFileDescription:
        samples = []
        df_shape = None
        df_columns = []
        try:
            df = pd.read_csv(data_file)
            if len(df) > 0:
                samples = df.head(max_samples).to_dict("records")
                samples = _truncate_long_values(samples)
            df_shape = df.shape
            df_columns = df.columns.tolist()
        except Exception as e:
            logger.warning(f"Error extracting samples from {data_file.name}: {e}")

        return FinetuneFileDescription(
            {"name": data_file.name, "type": "csv", "samples": samples, "shape": df_shape, "columns": df_columns}
        )

    def describe_file_parquet(self, data_file: Path, max_samples: int = 3) -> FinetuneFileDescription:
        samples = []
        df_shape = None
        df_columns = []
        try:
            df = pd.read_parquet(data_file)
            if len(df) > 0:
                samples = df.head(max_samples).to_dict("records")
                samples = _truncate_long_values(samples)
            df_shape = df.shape
            df_columns = df.columns.tolist()
        except Exception as e:
            logger.warning(f"Error extracting samples from {data_file.name}: {e}")

        return FinetuneFileDescription(
            {"name": data_file.name, "type": "parquet", "samples": samples, "shape": df_shape, "columns": df_columns}
        )

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


def _read_single_dataset_readme(dataset_path: Path, max_chars: int = 2000) -> str:
    """Read README file from a single dataset directory or its parent directories.

    Args:
        dataset_path: Path to the dataset directory
        max_chars: Maximum characters to read (default: 2000)

    Returns:
        README content as string, or empty string if not found
    """
    target_names = {"README.md", "readme.md", "README.txt", "README"}

    try:
        # Check current directory first
        for readme_name in target_names:
            readme_file = dataset_path / readme_name
            if readme_file.exists() and readme_file.is_file():
                try:
                    content = readme_file.read_text(encoding="utf-8")[:max_chars]
                    logger.info(f"Loaded README from {readme_file} ({len(content)} chars)")
                    return content
                except Exception as e:
                    logger.warning(f"Failed to read {readme_file}: {e}")

        # If not found in current directory, check parent directory
        parent_path = dataset_path.parent
        if parent_path != dataset_path:  # Avoid infinite loop at filesystem root
            for readme_name in target_names:
                readme_file = parent_path / readme_name
                if readme_file.exists() and readme_file.is_file():
                    try:
                        content = readme_file.read_text(encoding="utf-8")[:max_chars]
                        logger.info(f"Loaded README from parent directory {readme_file} ({len(content)} chars)")
                        return content
                    except Exception as e:
                        logger.warning(f"Failed to read {readme_file}: {e}")

        # If still not found, check one level down in subdirectories
        if dataset_path.exists():
            for item in dataset_path.iterdir():
                if item.is_dir():
                    for readme_name in target_names:
                        readme_file = item / readme_name
                        if readme_file.exists() and readme_file.is_file():
                            try:
                                content = readme_file.read_text(encoding="utf-8")[:max_chars]
                                logger.info(f"Loaded README from subdirectory {readme_file} ({len(content)} chars)")
                                return content
                            except Exception as e:
                                logger.warning(f"Failed to read {readme_file}: {e}")
    except Exception as e:
        logger.warning(f"Error searching for README in {dataset_path}: {e}")

    return ""


def check_all_dataset_in_info(ft_file_path, existing_config, max_depth: int = 3):
    """Scan datasets directory recursively and return dataset names not yet in existing_config.

    Recursively scans the datasets directory to find all directories containing data files.
    Supports configurable directory depth (default: 3 levels) and recognizes train/test/val split patterns.

    Smart split detection:
        - If a directory has subdirs like 'train', 'test', 'val' with data files,
          the parent directory is treated as the dataset
        - Otherwise, each directory with data files is a separate dataset

    Examples:
        - LIMO/limo.jsonl → dataset: "LIMO" (level 1)
        - s1K-1.1/data/train.parquet → dataset: "s1K-1.1/data" (level 2)
        - math/en/train/data.json + math/en/test/data.json → dataset: "math/en" (level 2)
        - code/python/file1.json + code/python/file2.json → dataset: "code/python" (level 2)

    Args:
        ft_file_path: Path to finetune directory structure
        existing_config: Existing dataset_info.json configuration
        max_depth: Maximum directory depth to scan (default: 3)

    Returns:
        list: Dataset names (relative paths) not yet in existing_config
    """
    root_path = Path(ft_file_path) / "datasets"
    dataset_list = []

    # Supported data file extensions
    data_extensions = {".json", ".jsonl", ".parquet", ".csv", ".arrow", ".txt"}

    # Common split names (train/test/validation patterns)
    split_names = {"train", "test", "val", "validation", "dev", "eval"}

    def has_data_files(directory: Path) -> bool:
        """Check if directory contains data files."""
        try:
            return any(f.is_file() and f.suffix in data_extensions for f in directory.iterdir())
        except:
            return False

    def scan_directory(current_path: Path, relative_path: str = "", depth: int = 0):
        """Recursively scan directory for data files."""
        # Check depth limit
        if depth >= max_depth:
            logger.debug(f"Reached max depth ({max_depth}) at {relative_path}, stopping scan")
            return

        try:
            # Get all items in current directory
            items = list(current_path.iterdir())

            # Check if current directory contains data files
            data_files = [f for f in items if f.is_file() and f.suffix in data_extensions]
            subdirs = [d for d in items if d.is_dir() and not d.name.startswith(".")]

            if data_files:
                # This directory contains data files directly, mark it as a dataset
                dataset_name = relative_path if relative_path else current_path.name
                dataset_list.append(dataset_name)
                # Don't recurse into subdirectories to avoid treating subsets as separate datasets
                return

            # Check if subdirectories look like train/test/val splits
            subdir_names = {d.name.lower() for d in subdirs}
            split_subdirs = subdir_names & split_names

            if split_subdirs and all(
                has_data_files(current_path / sd.name) for sd in subdirs if sd.name.lower() in split_names
            ):
                # This looks like a dataset with train/test/val splits
                # Mark the parent directory as the dataset
                dataset_name = relative_path if relative_path else current_path.name
                dataset_list.append(dataset_name)
                logger.info(f"Detected split dataset: {dataset_name} with splits: {split_subdirs}")
                return

            # If no data files and no split pattern, recurse into subdirectories
            for subdir in subdirs:
                new_relative_path = f"{relative_path}/{subdir.name}" if relative_path else subdir.name
                scan_directory(subdir, new_relative_path, depth + 1)

        except PermissionError:
            logger.warning(f"Permission denied accessing {current_path}")
        except Exception as e:
            logger.warning(f"Error scanning {current_path}: {e}")

    # Start scanning from root (depth 0 is the dataset root level)
    for item in root_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            scan_directory(item, item.name, depth=0)

    remain_dataset_list = [dataset_name for dataset_name in dataset_list if dataset_name not in existing_config]
    return remain_dataset_list


def get_dataset_folder_desc(ft_file_path: str, include_dataset_readme: bool = True) -> dict:
    """Get dataset folder description using AI analysis.

    Args:
        ft_file_path: Path to finetune directory structure

    Returns:
        dict: Dataset folder description
    """
    dataset_folder_desc = FinetuneDatasetDescriptor().describe_dataset_folder(
        Path(ft_file_path) / "datasets", include_dataset_readme=include_dataset_readme
    )
    return dataset_folder_desc


def generate_dataset_info_config(target_dataset_list: list, ft_file_path: str, existing_config: dict) -> dict:
    """Generate dataset_info.json configuration entry using AI for LLaMA-Factory compatibility.

    Args:
        target_dataset_list: List of specific datasets to process (empty for all)
        ft_file_path: Path to finetune directory structure
        existing_config: Existing dataset_info.json configuration

    Returns:
        dict: Configuration entry for dataset_info.json

    Raises:
        RuntimeError: If configuration generation or validation fails
    """
    # Use existing descriptor to get dataset information
    remain_dataset_list = check_all_dataset_in_info(ft_file_path, existing_config)
    if len(remain_dataset_list) == 0:
        return {}
    dataset_folder_desc = get_dataset_folder_desc(ft_file_path)
    real_target_dataset_list = remain_dataset_list if not target_dataset_list else target_dataset_list

    # Create prompt using template
    system_prompt = T(".prompts:dataset_info_generation.system").r(
        target_dataset_list=real_target_dataset_list,
    )
    user_prompt = T(".prompts:dataset_info_generation.user").r(
        dataset_info=str(dataset_folder_desc),
    )

    # Generate configuration using API
    api = APIBackend()
    raw_response = api.build_messages_and_create_chat_completion(
        system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True
    )

    response_dict = json.loads(raw_response)

    dataset_info_dict = {}
    datasets_root = Path(ft_file_path) / "datasets"

    for dataset_key, config in response_dict.items():
        if _validate_dataset_config(config):
            # Add README content for each dataset
            dataset_path = datasets_root / dataset_key
            if dataset_path.exists() and dataset_path.is_dir():
                readme_content = _read_single_dataset_readme(dataset_path, max_chars=5000)
                if readme_content:
                    config["readme"] = readme_content
                    logger.info(f"Added README to dataset '{dataset_key}' ({len(readme_content)} chars)")
                else:
                    logger.info(f"No README found for dataset '{dataset_key}'")

            # Log description status
            if "description" in config:
                logger.info(
                    f"LLM generated description for dataset '{dataset_key}' ({len(config['description'])} chars)"
                )
            else:
                logger.warning(f"No description generated for dataset '{dataset_key}'")

            dataset_info_dict[dataset_key] = config

    return dataset_info_dict


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
