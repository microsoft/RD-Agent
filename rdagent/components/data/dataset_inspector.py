"""Dataset inspection and basic quality checking without LLM calls."""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.scen.utils import preview_json
from rdagent.utils.agent.tpl import T


class DatasetInspector:
    """Inspect dataset structure and perform rule-based quality checks."""

    def __init__(self, api_backend: APIBackend | None = None):
        """
        Initialize inspector with optional LLM backend for file analysis.

        Args:
            api_backend: LLM API backend. If None, will create default instance.
        """
        self.api = api_backend or APIBackend()

    def inspect(self, dataset_path: str, sample_size: int = 100) -> dict[str, Any]:
        """
        Inspect dataset structure and sample data.

        Args:
            dataset_path: Path to the dataset directory
            sample_size: Number of samples to extract

        Returns:
            {
                "path": str,
                "loadable": bool,
                "total_samples": int,
                "columns": List[str],
                "sample_data": List[Dict],
                "files": List[str],
                "issues": List[str]
            }
        """
        result: dict[str, Any] = {
            "path": dataset_path,
            "loadable": False,
            "total_samples": 0,
            "columns": [],
            "sample_data": [],
            "files": [],
            "issues": [],
        }

        # Check if path exists
        if not os.path.exists(dataset_path):
            result["issues"].append(f"Path does not exist: {dataset_path}")
            return result

        # List files in directory
        try:
            result["files"] = os.listdir(dataset_path)
            logger.info(f"Dataset files: {result['files']}")
        except Exception as e:
            result["issues"].append(f"Failed to list directory: {e}")
            return result

        # Try to load dataset using HuggingFace datasets library
        try:
            # First try to load dataset normally
            dataset = load_dataset(dataset_path)
            result["loadable"] = True

            # Check if this is a multi-config dataset by examining the error
        except Exception as initial_error:
            error_msg = str(initial_error)
            if "Config name is missing" in error_msg:
                # This is a multi-config dataset, need to handle differently
                try:
                    # Get available configurations
                    available_configs = get_dataset_config_names(dataset_path)
                    if not available_configs:
                        raise ValueError(f"No configurations found for {dataset_path}")

                    logger.info(f"Multi-config dataset detected. Available configs: {available_configs}")
                    logger.info(f"Loading and merging ALL {len(available_configs)} configurations...")

                    # Load ALL configurations and merge them
                    all_splits = {}
                    total_samples = 0

                    for config in available_configs:
                        try:
                            config_dataset = load_dataset(dataset_path, config)

                            # Merge each split type (train, test, etc.)
                            for split_name in config_dataset.keys():
                                if split_name not in all_splits:
                                    all_splits[split_name] = []
                                all_splits[split_name].append(config_dataset[split_name])

                                if split_name == "train":
                                    total_samples += len(config_dataset[split_name])

                            logger.info(f"  ✓ Loaded {config}: {len(config_dataset.get('train', []))} train samples")

                        except Exception as e:
                            logger.warning(f"  ✗ Failed to load config {config}: {e}")

                    # Concatenate all splits
                    dataset = {}
                    for split_name, split_datasets in all_splits.items():
                        dataset[split_name] = concatenate_datasets(split_datasets)

                    result["loadable"] = True
                    logger.info("✅ Multi-config dataset merged successfully")
                    logger.info(f"   Total train samples: {total_samples} (from {len(available_configs)} configs)")

                    # Store config info for transparency
                    result["configs_merged"] = available_configs
                    result["total_configs"] = len(available_configs)

                except Exception as config_error:
                    result["issues"].append(f"Failed to handle multi-config dataset: {config_error}")
                    logger.error(f"❌ Multi-config dataset loading failed: {config_error}")
                    return result
            else:
                # Some other error occurred
                result["issues"].append(f"Failed to load dataset: {initial_error!s}")
                logger.error(f"❌ Failed to load dataset: {initial_error}")
                return result

        # Common processing for successfully loaded datasets
        if result["loadable"]:
            # Prefer 'train' split, otherwise use first available split
            if "train" in dataset:
                split = dataset["train"]
            else:
                split_name = list(dataset.keys())[0]
                split = dataset[split_name]
                result["issues"].append(f"No 'train' split found, using '{split_name}'")

            result["total_samples"] = len(split)
            result["columns"] = split.column_names

            # Sample first N rows
            sample_size = min(sample_size, len(split))
            samples = split.select(range(sample_size))
            result["sample_data"] = [samples[i] for i in range(sample_size)]

            logger.info(
                f"✅ Successfully loaded dataset: {result['total_samples']} samples, columns: {result['columns']}",
            )

        return result

    def check_quality(self, inspect_result: dict[str, Any]) -> dict[str, Any]:
        """
        Rule-based quality check without LLM calls.

        Args:
            inspect_result: Result from inspect() method

        Returns:
            {
                "is_usable": bool,
                "quality_score": float (0-10),
                "issues": List[str],
                "recommendations": List[str]
            }
        """
        issues = []
        score = 10.0  # Start with perfect score, deduct points for issues

        # Check 1: Is the dataset loadable?
        if not inspect_result["loadable"]:
            return {
                "is_usable": False,
                "quality_score": 0.0,
                "issues": ["Dataset cannot be loaded"],
                "recommendations": ["Check if dataset format is correct"],
            }

        # Check 2: Sample count
        total_samples = inspect_result["total_samples"]
        if total_samples < 100:
            issues.append(f"Too few samples: {total_samples}")
            score -= 3.0
        elif total_samples < 1000:
            issues.append(f"Sample count is low: {total_samples}")
            score -= 1.0

        # Check 3: Number of columns (too few may indicate incomplete structure)
        num_columns = len(inspect_result["columns"])
        if num_columns < 2:
            issues.append(f"Too few columns: {inspect_result['columns']}")
            score -= 2.0

        # Check 4: Check for empty/null values in first 10 samples
        empty_count = 0
        samples_to_check = inspect_result["sample_data"][:10]
        for sample in samples_to_check:
            for key, value in sample.items():
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    empty_count += 1

        if empty_count > 5:
            issues.append(f"Found {empty_count} empty/null fields in first 10 samples")
            score -= 1.0

        # Check 5: Check for duplicate column names
        columns = inspect_result["columns"]
        if len(columns) != len(set(columns)):
            issues.append("Duplicate column names detected")
            score -= 2.0

        # Ensure score doesn't go below 0
        score = max(0.0, score)

        # Determine usability
        is_usable = score >= 5.0

        # Generate recommendations
        recommendations = []
        if not is_usable:
            recommendations.append("Recommend discarding this dataset")
        elif score < 7.0:
            recommendations.append("Recommend manual review before use")
        else:
            recommendations.append("Quality looks good, can be migrated")

        return {
            "is_usable": is_usable,
            "quality_score": score,
            "issues": issues,
            "recommendations": recommendations,
        }

    def analyze_files_for_sft(
        self,
        dataset_path: str,
        task_description: str,
    ) -> dict[str, Any]:
        """
        Use LLM to analyze which files are useful for SFT training.

        Args:
            dataset_path: Path to the dataset directory
            task_description: Original task description (e.g., "数学推理数据集")

        Returns:
            {
                "useful_files": List[str],      # Files to keep
                "junk_files": List[str],        # Files to discard
                "file_analysis": Dict,           # Detailed analysis per file
                "total_size_mb": float,
                "size_after_cleanup_mb": float,
                "space_saved_mb": float
            }
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        # Collect all files with sizes
        all_files = []
        total_size_bytes = 0

        for root, dirs, files in os.walk(dataset_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if file.startswith("."):
                    continue

                file_path = Path(root) / file
                file_size_bytes = file_path.stat().st_size
                total_size_bytes += file_size_bytes

                # Relative path from dataset root
                rel_path = file_path.relative_to(dataset_path)

                all_files.append(
                    {
                        "name": str(rel_path),
                        "size_mb": round(file_size_bytes / (1024 * 1024), 2),
                        "extension": file_path.suffix,
                    },
                )

        logger.info(f"Found {len(all_files)} files, total size: {total_size_bytes / (1024 * 1024):.2f}MB")

        # Extract comprehensive file contents (markdown docs + data previews)
        file_contents = self._extract_all_file_contents(dataset_path)
        logger.info(f"Extracted file contents ({len(file_contents)} chars)")

        # Call LLM to analyze files
        sys_prompt = T(".prompts:analyze_files_for_sft.system").r()
        user_prompt = T(".prompts:analyze_files_for_sft.user").r(
            task_description=task_description,
            files_json=json.dumps(all_files, ensure_ascii=False, indent=2),
            file_contents=file_contents,
        )

        try:
            response = self.api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                json_mode=True,
                json_target_type=dict,
            )

            # Parse JSON if response is string
            if isinstance(response, str):
                response = json.loads(response)

            # Validate response
            if not isinstance(response, dict) or "file_analysis" not in response:
                raise ValueError(f"Invalid LLM response format: {response}")

            file_analysis = response["file_analysis"]

            # Categorize files
            useful_files = []
            junk_files = []
            size_after_cleanup_bytes = 0

            for file_info in all_files:
                file_name = file_info["name"]
                file_size_bytes = file_info["size_mb"] * 1024 * 1024

                # LLM may not analyze all files, default to useful
                llm_analysis = file_analysis.get(file_name, {"useful": True, "reason": "Not analyzed by LLM"})

                if llm_analysis.get("useful", True):
                    useful_files.append(file_name)
                    size_after_cleanup_bytes += file_size_bytes
                else:
                    junk_files.append(file_name)

            total_size_mb = total_size_bytes / (1024 * 1024)
            size_after_cleanup_mb = size_after_cleanup_bytes / (1024 * 1024)
            space_saved_mb = total_size_mb - size_after_cleanup_mb

            logger.info(f"Analysis complete: {len(useful_files)} useful, {len(junk_files)} junk")
            logger.info(f"Space saved: {space_saved_mb:.2f}MB")

            return {
                "useful_files": useful_files,
                "junk_files": junk_files,
                "file_analysis": file_analysis,
                "total_size_mb": round(total_size_mb, 2),
                "size_after_cleanup_mb": round(size_after_cleanup_mb, 2),
                "space_saved_mb": round(space_saved_mb, 2),
            }

        except Exception as e:
            logger.error(f"Failed to analyze files with LLM: {e}")
            # Fallback: simple rule-based filtering
            logger.warning("Using rule-based fallback for file filtering")
            return self._fallback_file_analysis(all_files, total_size_bytes)

    def _extract_all_markdown_docs(self, dataset_path: Path) -> str:
        """
        Extract content from all markdown files in the dataset directory.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Formatted string containing all markdown file contents
        """
        markdown_contents = []

        # Find all .md files recursively
        for md_file in dataset_path.rglob("*.md"):
            if md_file.is_file():
                try:
                    # Read the markdown content
                    content = md_file.read_text(encoding="utf-8", errors="ignore")

                    # Get relative path from dataset root
                    rel_path = md_file.relative_to(dataset_path)

                    # Limit content length to prevent prompt overflow (max 2000 chars per file)
                    if len(content) > 2000:
                        content = content[:2000] + "\n... (content truncated)"

                    # Format as: ### filename.md\ncontent\n
                    markdown_section = f"### {rel_path}\n\n{content}\n"
                    markdown_contents.append(markdown_section)

                except Exception as e:
                    logger.warning(f"Failed to read markdown file {md_file}: {e}")

        if not markdown_contents:
            return "No markdown documents found"

        # Combine all markdown contents
        result = "## Markdown Documentation\n\n" + "\n".join(markdown_contents)

        return result

    def _extract_data_file_previews(self, dataset_path: Path) -> str:
        """
        Extract previews of data files to show actual content structure.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Formatted string containing data file previews
        """
        data_previews = []

        # Define data file patterns and their preview methods
        data_patterns = {
            ".csv": self._preview_csv_file,
            ".json": self._preview_json_file,
            ".jsonl": self._preview_jsonl_file,
            ".parquet": self._preview_parquet_file,
            ".arrow": self._preview_arrow_file,
            ".txt": self._preview_text_file,
        }

        # Find data files and generate previews
        for file_path in dataset_path.rglob("*"):
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                file_size_mb = file_path.stat().st_size / (1024 * 1024)

                # Skip large files (>10MB) to avoid memory issues
                if file_size_mb > 10:
                    continue

                if file_ext in data_patterns:
                    try:
                        rel_path = file_path.relative_to(dataset_path)
                        preview_func = data_patterns[file_ext]
                        # Pass relative path to preview function
                        # Note: preview functions already include file name as header
                        preview = preview_func(file_path, rel_path)

                        # Add file size info to the header line
                        # Replace the header line to include file size
                        preview_lines = preview.split("\n")
                        if preview_lines and preview_lines[0].startswith("###"):
                            # Append file size to the header line
                            preview_lines[0] = f"{preview_lines[0].rstrip(':')} ({file_size_mb:.2f}MB):"
                            preview = "\n".join(preview_lines)

                        data_previews.append(preview)

                    except Exception as e:
                        logger.warning(f"Failed to preview {file_path}: {e}")

        if not data_previews:
            return "No data file previews found"

        # Join previews with double newline for separation between files
        result = "## Data File Previews\n\n" + "\n\n".join(data_previews)

        return result

    def _format_full_dataframe_preview(self, df, rel_path: Path, n_rows: int = 3) -> str:
        """
        Format DataFrame preview showing first N rows with FULL content in Key-Value format.

        Args:
            df: DataFrame to preview
            rel_path: Relative file path
            n_rows: Number of rows to show (default 3)

        Returns:
            Formatted preview string with complete data in Key-Value format (LLM-friendly)
        """
        preview_lines = [
            f"### {rel_path}:",
            "#### 1.DataFrame preview:",
            f"It has {df.shape[0]} rows and {df.shape[1]} columns.",
            f"The columns are: {', '.join(df.columns)}",
            f"#### 2.DataFrame preview: (first {n_rows} rows, FULL content)",
            "",
        ]

        # Format each row as Key-Value structure (LLM-friendly)
        for idx in range(min(n_rows, len(df))):
            row = df.iloc[idx]
            preview_lines.append(f"Row {idx}:")
            for col_name, value in row.items():
                # Convert value to string, handle NaN/None
                value_str = str(value) if value is not None else "None"
                preview_lines.append(f"  - {col_name}: {value_str}")
            preview_lines.append("")  # Blank line between rows

        return "\n".join(preview_lines)

    def _preview_csv_file(self, file_path: Path, rel_path: Path) -> str:
        """Preview CSV file content with full data (first 3 rows, no truncation)."""

        try:
            df = pd.read_csv(file_path)
            return self._format_full_dataframe_preview(df, rel_path, n_rows=3)
        except Exception as e:
            logger.warning(f"Failed to preview CSV {file_path}: {e}")
            return f"### {rel_path}:\nCSV file (preview failed: {str(e)[:50]})"

    def _preview_json_file(self, file_path: Path, rel_path: Path) -> str:
        """Preview JSON file with full data (first 3 rows if tabular, no truncation)."""

        try:
            # Try loading as DataFrame first (if JSON is tabular)
            df = pd.read_json(file_path)
            return self._format_full_dataframe_preview(df, rel_path, n_rows=3)
        except Exception:
            # Fallback: use original preview_json if not tabular
            try:
                preview = preview_json(file_path, str(rel_path))
                return preview
            except Exception as e:
                logger.warning(f"Failed to preview JSON {file_path}: {e}")
                return f"### {rel_path}:\nJSON file (preview failed: {str(e)[:50]})"

    def _preview_jsonl_file(self, file_path: Path, rel_path: Path) -> str:
        """Preview JSONL file with full data (first 3 rows if tabular, no truncation)."""

        try:
            # JSONL can be loaded as DataFrame with lines=True
            df = pd.read_json(file_path, lines=True)
            return self._format_full_dataframe_preview(df, rel_path, n_rows=3)
        except Exception:
            # Fallback: use original preview_json if not tabular
            try:
                preview = preview_json(file_path, str(rel_path))
                return preview
            except Exception as e:
                logger.warning(f"Failed to preview JSONL {file_path}: {e}")
                return f"### {rel_path}:\nJSONL file (preview failed: {str(e)[:50]})"

    def _preview_parquet_file(self, file_path: Path, rel_path: Path) -> str:
        """Preview parquet file with full data (first 3 rows, no truncation)."""

        try:
            df = pd.read_parquet(file_path)
            return self._format_full_dataframe_preview(df, rel_path, n_rows=3)
        except Exception as e:
            logger.warning(f"Failed to preview Parquet {file_path}: {e}")
            return f"### {rel_path}:\nParquet file (preview failed: {str(e)[:50]})"

    def _preview_arrow_file(self, file_path: Path, rel_path: Path) -> str:
        """Preview arrow file - just show file info since it's binary."""
        try:
            file_size_kb = file_path.stat().st_size / 1024
            return f"Arrow file\nSize: {file_size_kb:.1f}KB\nFormat: Apache Arrow binary format"
        except Exception:
            return "Arrow file (cannot preview binary format)"

    def _preview_text_file(self, file_path: Path, rel_path: Path) -> str:
        """Preview text file content."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            # Show first 1000 chars
            if len(content) > 1000:
                preview = f"```\n{content[:1000]}...\n```"
            else:
                preview = f"```\n{content}\n```"
            return preview + f"\nFormat: Plain text file, {len(content)} characters"
        except Exception:
            return "Cannot read text content"

    def _extract_all_file_contents(self, dataset_path: Path) -> str:
        """
        Extract both markdown docs and data file previews.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Combined formatted string with all file contents
        """
        markdown_content = self._extract_all_markdown_docs(dataset_path)
        data_previews = self._extract_data_file_previews(dataset_path)

        # Combine both sections
        combined_content = f"{markdown_content}\n\n{data_previews}"

        return combined_content

    def _fallback_file_analysis(self, all_files: list[dict], total_size_bytes: int) -> dict[str, Any]:
        """
        Fallback rule-based file filtering when LLM fails.

        Keeps: .csv, .json, .jsonl, .parquet, .arrow
        Removes: .zip, .tar.gz, .cache, .git*, README.md
        """
        KEEP_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet", ".arrow"}
        JUNK_PATTERNS = {".cache", ".git", "readme.md"}

        useful_files = []
        junk_files = []
        file_analysis = {}
        size_after_cleanup_bytes = 0

        for file_info in all_files:
            file_name = file_info["name"]
            file_ext = file_info["extension"].lower()
            file_name_lower = file_name.lower()
            file_size_bytes = file_info["size_mb"] * 1024 * 1024

            is_useful = False
            reason = ""

            # Check if junk
            if any(pattern in file_name_lower for pattern in JUNK_PATTERNS):
                is_useful = False
                reason = "Junk file (cache/git/readme)"
            # Check extension
            elif file_ext in KEEP_EXTENSIONS:
                is_useful = True
                reason = "Data file with valid extension"
            # Compressed files
            elif file_ext in {".zip", ".tar", ".gz"}:
                is_useful = False
                reason = "Compressed file (likely images/audio)"
            # Data directory
            elif "data/" in file_name:
                is_useful = True
                reason = "File in data directory"
            # Default: keep unknown files to be safe
            else:
                is_useful = True
                reason = "Unknown file type, keeping to be safe"

            file_analysis[file_name] = {"useful": is_useful, "reason": reason}

            if is_useful:
                useful_files.append(file_name)
                size_after_cleanup_bytes += file_size_bytes
            else:
                junk_files.append(file_name)

        total_size_mb = total_size_bytes / (1024 * 1024)
        size_after_cleanup_mb = size_after_cleanup_bytes / (1024 * 1024)
        space_saved_mb = total_size_mb - size_after_cleanup_mb

        return {
            "useful_files": useful_files,
            "junk_files": junk_files,
            "file_analysis": file_analysis,
            "total_size_mb": round(total_size_mb, 2),
            "size_after_cleanup_mb": round(size_after_cleanup_mb, 2),
            "space_saved_mb": round(space_saved_mb, 2),
        }
