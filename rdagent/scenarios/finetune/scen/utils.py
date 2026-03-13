"""Utilities for fine-tuning scenario data extraction and analysis."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tiktoken

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.scen.utils import FileTreeGenerator
from rdagent.utils import md5_hash

# Fixed tokenizer model for token counting
_TOKENIZER_MODEL = "gpt-3.5-turbo"


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
    """Recursively truncate long string values in nested data structures.

    Args:
        obj: The object to truncate (dict, list, ndarray, or str)
        max_length: Maximum length for string values

    Returns:
        Truncated object with the same structure, showing omitted character count.
        numpy arrays are converted to Python lists for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        # Convert numpy array to list first, then process recursively
        return _truncate_long_values(obj.tolist(), max_length)
    elif isinstance(obj, dict):
        return {k: _truncate_long_values(v, max_length) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_truncate_long_values(item, max_length) for item in obj]
    elif isinstance(obj, str) and len(obj) > max_length:
        omitted = len(obj) - max_length
        return obj[:max_length] + f"...(omitted {omitted} chars)"
    elif isinstance(obj, (np.integer, np.floating)):
        # Convert numpy scalar types to Python native types
        return obj.item()
    return obj


def _compute_column_stats(data: list[dict]) -> dict[str, dict]:
    """Compute token statistics for each string column in the dataset.

    Uses tiktoken batch encoding for 10-50x faster processing.
    Fixed to use gpt-3.5-turbo tokenizer.

    Args:
        data: List of dictionaries representing dataset samples

    Returns:
        Dictionary mapping column names to their token statistics:
        {column_name: {empty_count, min_tokens, max_tokens, p50_tokens, p99_tokens}}
    """
    if not data:
        return {}

    # Collect all column names from the dataset
    all_columns: set[str] = set()
    for item in data:
        if isinstance(item, dict):
            all_columns.update(item.keys())

    # Get tiktoken encoder (cached after first call)
    try:
        encoding = tiktoken.encoding_for_model(_TOKENIZER_MODEL)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    column_stats = {}
    for col in all_columns:
        texts: list[str] = []
        empty_count = 0

        # Collect all non-empty texts for this column
        for item in data:
            if isinstance(item, dict):
                val = item.get(col, "")
                if isinstance(val, str):
                    if not val.strip():
                        empty_count += 1
                    else:
                        texts.append(val)

        if texts:
            # Batch encode all texts at once (10-50x faster than individual calls)
            try:
                encoded_batch = encoding.encode_batch(texts)
                token_counts = [len(tokens) for tokens in encoded_batch]
            except Exception as e:
                logger.warning(f"Batch encoding failed for column '{col}': {e}, falling back to sequential")
                token_counts = [len(encoding.encode(t)) for t in texts]

            column_stats[col] = {
                "empty_count": empty_count,
                "min_tokens": int(min(token_counts)),
                "max_tokens": int(max(token_counts)),
                "p50_tokens": int(np.percentile(token_counts, 50)),
                "p99_tokens": int(np.percentile(token_counts, 99)),
            }
        else:
            column_stats[col] = {
                "empty_count": empty_count,
                "min_tokens": 0,
                "max_tokens": 0,
                "p50_tokens": 0,
                "p99_tokens": 0,
            }

    return column_stats


def _load_dataset_for_stats(data_files: list[Path], max_samples: int = 50000) -> list[dict]:
    """Load dataset samples from data files for statistics computation.

    Args:
        data_files: List of data file paths
        max_samples: Maximum number of samples to load

    Returns:
        List of dictionaries representing dataset samples
    """
    all_data: list[dict] = []

    for data_file in data_files:
        if len(all_data) >= max_samples:
            break

        suffix = data_file.suffix.lower()
        try:
            if suffix == ".json":
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data[: max_samples - len(all_data)])
                    elif isinstance(data, dict):
                        all_data.append(data)

            elif suffix == ".jsonl":
                with open(data_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(all_data) >= max_samples:
                            break
                        line = line.strip()
                        if line:
                            all_data.append(json.loads(line))

            elif suffix == ".csv":
                df = pd.read_csv(data_file, nrows=max_samples - len(all_data))
                all_data.extend(df.to_dict("records"))

            elif suffix == ".parquet":
                df = pd.read_parquet(data_file)
                all_data.extend(df.head(max_samples - len(all_data)).to_dict("records"))

        except Exception as e:
            logger.warning(f"Failed to load {data_file.name} for stats: {e}")

    return all_data


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

    def _generate_stats(self, dataset_path: Path, include_column_stats: bool = False) -> dict[str, Any]:
        """Calculate dataset statistics: sample count, file size, and optionally column token stats.

        Args:
            dataset_path: Path to the dataset directory
            include_column_stats: Whether to compute per-column token statistics

        Returns:
            Dictionary with sample_count, total_size_mb, file_count, and optionally column_stats.
            Note: column_stats contains TOKEN counts (not character lengths) for each column,
            using gpt-3.5-turbo tokenizer:
            {column_name: {empty_count, min_tokens, max_tokens, p50_tokens, p99_tokens}}
        """
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

            stats = {
                "sample_count": total_samples,
                "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
                "file_count": file_count,
            }

            # Compute column token statistics if requested
            if include_column_stats and data_files:
                try:
                    dataset_samples = _load_dataset_for_stats(data_files)
                    if dataset_samples:
                        stats["column_stats"] = _compute_column_stats(dataset_samples)
                        logger.info(
                            f"Computed column token stats for {len(stats['column_stats'])} columns "
                            f"(using tokenizer: {_TOKENIZER_MODEL})"
                        )
                except Exception as e:
                    logger.warning(f"Failed to compute column token stats: {e}")

            return stats

        except Exception as e:
            logger.warning(f"Failed to calculate dataset stats: {e}")
            return {
                "sample_count": 0,
                "total_size_mb": 0,
                "file_count": 0,
            }

    def hash_dataset_path(
        self, dataset_path: Path, dataset_name: str | None = None, include_dataset_readme: bool = False
    ) -> str:
        """Generate hash key for dataset description caching."""
        key_parts = []
        key_parts.append(str(dataset_path))
        files = sorted(str(path.relative_to(dataset_path)) for path in dataset_path.rglob("*") if path.is_file())
        key_parts.append(",".join(files))
        if dataset_name:
            key_parts.append(dataset_name)
        key_parts.append(str(include_dataset_readme))
        return md5_hash("|".join(key_parts))

    @cache_with_pickle(hash_dataset_path)
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
            logger.info(f"Generating dataset folder description for {dataset_path}...")
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

        # Check if tokenizer supports <think> token for CoT training
        info["has_think_token"] = False
        tokenizer_path = model_path / "tokenizer.json"
        if tokenizer_path.exists():
            try:
                with open(tokenizer_path, encoding="utf-8") as f:
                    tokenizer_config = json.load(f)
                    # Check in vocabulary
                    vocab = tokenizer_config.get("model", {}).get("vocab", {})
                    # Check in added_tokens
                    added_tokens = tokenizer_config.get("added_tokens", [])
                    added_token_contents = {t.get("content") for t in added_tokens if isinstance(t, dict)}

                    if "<think>" in vocab or "<think>" in added_token_contents:
                        info["has_think_token"] = True
                        logger.info(f"Model {model_name} has native <think> token support")
            except Exception as e:
                logger.warning(f"Failed to check tokenizer for <think> token: {e}")

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

    def _discover_subtasks(self, dataset_dir: Path) -> dict:
        """Discover subtasks by scanning directory structure.

        Groups data files by their parent directory name. The deepest directory
        containing data files is considered a subtask.

        Args:
            dataset_dir: Root directory of the dataset

        Returns:
            Dictionary mapping subtask names to their info:
            {subtask_name: {"files": [relative_paths], "file_paths": [absolute_paths]}}
        """
        data_extensions = {".json", ".jsonl", ".parquet", ".csv"}
        subtasks: dict[str, dict] = {}

        for data_file in dataset_dir.rglob("*"):
            if not data_file.is_file():
                continue
            if data_file.suffix.lower() not in data_extensions:
                continue
            if data_file.name.startswith("."):
                continue

            rel_path = data_file.relative_to(dataset_dir)
            # Use deepest directory name as subtask, or "_root" if file is in top-level
            subtask_name = rel_path.parent.name if len(rel_path.parts) > 1 else "_root"

            if subtask_name not in subtasks:
                subtasks[subtask_name] = {"files": [], "file_paths": []}
            subtasks[subtask_name]["files"].append(str(rel_path))
            subtasks[subtask_name]["file_paths"].append(data_file)

        return subtasks

    def analyze_dataset(self, dataset_dir: Path) -> dict:
        """Analyze a dataset directory and generate dataset_info.json entry.

        This method:
        1. Reads README from the dataset directory
        2. Generates file tree for LLM understanding
        3. Discovers tasks by directory structure
        4. Computes statistics for each task (sample count, token stats)
        5. Extracts sample data for each task

        All datasets have a unified "tasks" structure. For datasets with files
        directly in the root directory, "_root" is used as the task name.

        Args:
            dataset_dir: Root directory of the dataset

        Returns:
            Dictionary containing dataset info ready for dataset_info.json
        """
        # 1. Read README
        readme = self._read_dataset_readme(dataset_dir)

        # 2. Generate file tree (for LLM to understand directory structure)
        file_tree = self._generate_file_tree(dataset_dir)

        # 3. Discover tasks
        tasks = self._discover_subtasks(dataset_dir)

        if not tasks:
            logger.warning(f"No data files found in {dataset_dir}")
            return {
                "readme": readme,
                "file_tree": file_tree,
                "total_samples": 0,
                "total_size_mb": 0,
                "tasks": {},
            }

        # 4. Compute stats for each task
        total_samples = 0
        total_size = 0
        for name, info in tasks.items():
            file_paths = info["file_paths"]
            data = _load_dataset_for_stats(file_paths)
            info["sample_count"] = len(data)
            info["column_stats"] = _compute_column_stats(data)
            info["samples"] = _truncate_long_values(self._extract_samples_for_template(file_paths, max_samples=3))
            total_samples += info["sample_count"]
            total_size += sum(f.stat().st_size for f in file_paths)
            # Remove file_paths as it's not JSON serializable and not needed in output
            del info["file_paths"]

        # 5. Return unified structure (all datasets have tasks)
        return {
            "readme": readme,
            "file_tree": file_tree,
            "total_samples": total_samples,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "tasks": tasks,
        }


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
    """Scan datasets directory and return top-level dataset names not yet in existing_config.

    Only scans first-level directories under datasets/. Each top-level directory is treated
    as a single dataset, regardless of its internal structure.

    Examples:
        - datasets/chemcot/ → dataset: "chemcot"
        - datasets/panorama/ → dataset: "panorama"
        - datasets/deepscaler/ → dataset: "deepscaler"

    Args:
        ft_file_path: Path to finetune directory structure
        existing_config: Existing dataset_info.json configuration
        max_depth: Unused, kept for API compatibility

    Returns:
        list: Dataset names (top-level directory names) not yet in existing_config
    """
    root_path = Path(ft_file_path) / "datasets"
    dataset_list = []

    try:
        for item in root_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                dataset_list.append(item.name)
    except Exception as e:
        logger.warning(f"Error scanning datasets directory: {e}")

    remain_dataset_list = [dataset_name for dataset_name in dataset_list if dataset_name not in existing_config]
    return remain_dataset_list


def generate_dataset_info_config(target_dataset_list: list, ft_file_path: str, existing_config: dict) -> dict:
    """Generate dataset_info.json configuration with auto-discovered subtasks.

    This function analyzes datasets not yet in existing_config and generates
    structured information including:
    - README content
    - File tree structure
    - Auto-discovered subtasks with statistics
    - Column token statistics for each subtask
    - Sample data for LLM understanding

    The dataset_info.json acts as a cache - existing datasets are skipped.

    Args:
        target_dataset_list: List of specific datasets to process (empty for all)
        ft_file_path: Path to finetune directory structure
        existing_config: Existing dataset_info.json configuration (used as cache)

    Returns:
        dict: New configuration entries for dataset_info.json
    """
    # Find datasets not yet in existing_config
    remain_dataset_list = check_all_dataset_in_info(ft_file_path, existing_config)
    if not remain_dataset_list:
        return {}

    datasets_root = Path(ft_file_path) / "datasets"
    descriptor = FinetuneDatasetDescriptor()
    new_config = {}

    # Determine which datasets to process
    datasets_to_process = (
        remain_dataset_list if not target_dataset_list else [d for d in target_dataset_list if d in remain_dataset_list]
    )

    for dataset_name in datasets_to_process:
        dataset_dir = datasets_root / dataset_name
        if dataset_dir.exists() and dataset_dir.is_dir():
            logger.info(f"Analyzing dataset '{dataset_name}'...")
            new_config[dataset_name] = descriptor.analyze_dataset(dataset_dir)
            logger.info(
                f"Analyzed dataset '{dataset_name}': "
                f"{new_config[dataset_name].get('total_samples', 0)} samples, "
                f"{new_config[dataset_name].get('total_size_mb', 0)} MB"
            )

    return new_config
