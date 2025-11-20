"""Data converter for SFT dataset preparation.

Converts datasets to Alpaca format for SFT/LoRA training.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class DataConverter:
    """Convert datasets to Alpaca format."""

    # Metadata field patterns (for automatic metadata extraction)
    METADATA_BLACKLIST_PATTERNS = {
        # Internal identifiers (should not be preserved)
        "id",
        "idx",
        "index",
        "row_id",
        "record_id",
        "sample_id",
        # Timestamps (usually not useful for training)
        "timestamp",
        "date",
        "time",
        "created_at",
        "updated_at",
        # Alpaca core fields (already extracted separately)
        "instruction",
        "input",
        "output",
        # Technical metadata (internal use only)
        "file_path",
        "source_file",
        "url",
        "hash",
        # Reserved keywords
        "_id",
        "__id",
        "metadata",
        "meta",
    }

    METADATA_WHITELIST_PATTERNS = {
        # Explicitly preserve these fields (even if match blacklist)
        "category",
        "type",
        "level",
        "difficulty",
        "domain",
        "subject",
        "topic",
        "tag",
        "label",
        "language",
        "quality",
        "source",
        "version",
        "license",
    }

    def __init__(self):
        """Initialize data converter."""

    def convert_to_alpaca(
        self,
        data: pd.DataFrame,
        schema: dict[str, Any],
        enable_metadata: bool = True,
    ) -> list[dict[str, str]]:
        """Convert dataset to Alpaca format.

        Args:
            data: DataFrame to convert
            schema: Schema analysis result from SchemaAnalyzer
            enable_metadata: Whether to extract and preserve metadata fields (default: True)

        Returns:
            List of Alpaca-format samples:
            [
                {
                    "instruction": "...",
                    "input": "...",
                    "output": "...",
                    "metadata": {...}  # Optional, only if enable_metadata=True and metadata exists
                }
            ]
        """
        data_type = schema["data_type"]
        instruction_col = schema["instruction_col"]
        input_col = schema.get("input_col")
        output_col = schema["output_col"]

        logger.info(
            f"Converting {len(data)} samples to Alpaca format "
            f"(data_type={data_type}, instruction={instruction_col}, "
            f"input={input_col}, output={output_col}, enable_metadata={enable_metadata})",
        )

        if data_type == "single_turn":
            return self._convert_single_turn(data, instruction_col, input_col, output_col, enable_metadata)
        if data_type == "multi_turn":
            return self._convert_multi_turn(data, instruction_col, enable_metadata)
        raise ValueError(f"Unknown data_type: {data_type}")

    def _convert_single_turn(
        self,
        data: pd.DataFrame,
        instruction_col: str,
        input_col: str | None,
        output_col: str,
        enable_metadata: bool = True,
    ) -> list[dict[str, str]]:
        """Convert single-turn Q&A data to Alpaca format.

        Args:
            data: DataFrame
            instruction_col: Column name for instruction
            input_col: Column name for input (optional)
            output_col: Column name for output
            enable_metadata: Whether to extract and preserve metadata fields (default: True)

        Returns:
            List of Alpaca-format samples
        """
        alpaca_samples = []

        # Build core columns set for metadata extraction
        core_cols = {instruction_col, output_col}
        if input_col:
            core_cols.add(input_col)

        for idx, row in data.iterrows():
            try:
                # Extract fields
                instruction = self._extract_field(row, instruction_col)
                output = self._extract_field(row, output_col)

                # Extract input (optional)
                if input_col:
                    input_text = self._extract_field(row, input_col)
                else:
                    input_text = ""

                # Validate required fields
                if not instruction or not output:
                    logger.debug(
                        f"Skipping row {idx}: empty instruction or output "
                        f"(instruction={bool(instruction)}, output={bool(output)})",
                    )
                    continue

                # Build Alpaca sample
                alpaca_sample = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                }

                # Extract metadata (optional)
                metadata = self._extract_metadata(row, core_cols, enable_metadata)
                if metadata:
                    alpaca_sample["metadata"] = metadata

                alpaca_samples.append(alpaca_sample)

            except Exception as e:
                logger.warning(f"Failed to convert row {idx}: {e}")
                continue

        logger.info(f"Converted {len(alpaca_samples)}/{len(data)} samples successfully")
        return alpaca_samples

    def _extract_field(self, row: pd.Series, col_name: str) -> str:
        """Extract and clean field value from row.

        Args:
            row: DataFrame row
            col_name: Column name

        Returns:
            Cleaned string value
        """
        value = row[col_name]

        # Handle NaN/None
        if pd.isna(value):
            return ""

        # Convert to string
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value).strip()
        if isinstance(value, (list, dict)):
            # For complex types, convert to JSON string
            return json.dumps(value, ensure_ascii=False)
        return str(value).strip()

    def _is_metadata_field(self, col_name: str) -> bool:
        """Check if a field should be included in metadata.

        Uses whitelist-first approach:
        1. If in whitelist → include (highest priority)
        2. If in blacklist → exclude
        3. Otherwise → include (default behavior)

        Args:
            col_name: Column name to check

        Returns:
            True if field should be preserved in metadata, False otherwise
        """
        col_lower = col_name.lower()

        # Whitelist takes priority (always include)
        if any(pattern in col_lower for pattern in self.METADATA_WHITELIST_PATTERNS):
            return True

        # Blacklist (exclude)
        if any(pattern == col_lower or pattern in col_lower for pattern in self.METADATA_BLACKLIST_PATTERNS):
            return False

        # Default: include unknown fields (conservative approach)
        return True

    def _extract_metadata(
        self,
        row: pd.Series,
        core_cols: set,
        enable_metadata: bool = True,
    ) -> dict[str, Any]:
        """Extract metadata from row, excluding core columns.

        Metadata includes non-core fields that provide additional context
        (e.g., category, type, level, difficulty, source).

        Args:
            row: DataFrame row
            core_cols: Set of core column names to exclude (instruction/input/output)
            enable_metadata: Whether to extract metadata (default: True)

        Returns:
            Dict of metadata fields. Empty dict if enable_metadata=False or no metadata found.
        """
        if not enable_metadata:
            return {}

        metadata = {}

        for col in row.index:
            # Skip core columns (already extracted separately)
            if col in core_cols:
                continue

            # Check if this field should be included in metadata
            if not self._is_metadata_field(col):
                continue

            # Extract value
            value = self._extract_field(row, col)

            # Only include non-empty values
            if value:
                metadata[col] = value

        return metadata

    def _convert_multi_turn(
        self,
        data: pd.DataFrame,
        conversation_col: str,
        enable_metadata: bool = True,
    ) -> list[dict[str, str]]:
        """Convert multi-turn dialogue data to Alpaca format.

        Strategy: Preserve conversation history as input field.
        Each turn (after the first) includes previous conversation as context.

        Args:
            data: DataFrame
            conversation_col: Column containing conversation data
            enable_metadata: Whether to extract and preserve metadata fields (default: True)

        Returns:
            List of Alpaca-format samples
        """
        alpaca_samples = []

        # Build core columns set for metadata extraction
        core_cols = {conversation_col}

        for idx, row in data.iterrows():
            try:
                conversation = row[conversation_col]

                # Extract conversation-level metadata (same for all turns in this conversation)
                conversation_metadata = self._extract_metadata(row, core_cols, enable_metadata)

                # Parse conversation structure
                turns = self._parse_conversation(conversation)

                if not turns:
                    logger.debug(f"Skipping row {idx}: empty conversation")
                    continue

                # Convert each turn to Alpaca format
                conversation_history = []
                for turn_idx, turn in enumerate(turns):
                    user_msg = turn.get("user", "")
                    assistant_msg = turn.get("assistant", "")

                    if not user_msg or not assistant_msg:
                        logger.debug(f"Skipping turn {turn_idx} in row {idx}: empty user or assistant message")
                        continue

                    # Build input context from previous conversation
                    if conversation_history:
                        input_context = self._format_conversation_history(conversation_history)
                    else:
                        input_context = ""

                    # Build Alpaca sample
                    alpaca_sample = {
                        "instruction": user_msg,
                        "input": input_context,
                        "output": assistant_msg,
                    }

                    # Add conversation-level metadata to each turn
                    if conversation_metadata:
                        alpaca_sample["metadata"] = conversation_metadata

                    alpaca_samples.append(alpaca_sample)

                    # Update conversation history
                    conversation_history.append({"user": user_msg, "assistant": assistant_msg})

            except Exception as e:
                logger.warning(f"Failed to convert multi-turn conversation in row {idx}: {e}")
                continue

        logger.info(f"Converted {len(alpaca_samples)} turns from {len(data)} conversations")
        return alpaca_samples

    def _parse_conversation(self, conversation: Any) -> list[dict[str, str]]:
        """Parse conversation data into structured turns.

        Supports multiple formats:
        - OpenAI: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        - ShareGPT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        - Custom formats

        Args:
            conversation: Conversation data (list or dict)

        Returns:
            List of turns: [{"user": "...", "assistant": "..."}, ...]
        """
        # Handle NaN/None
        if pd.isna(conversation):
            return []

        # Parse string as JSON if needed
        if isinstance(conversation, str):
            try:
                conversation = json.loads(conversation)
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse conversation string as JSON: {conversation[:100]}")
                return []

        # Extract messages list
        messages = []
        if isinstance(conversation, list):
            messages = conversation
        elif isinstance(conversation, dict):
            # Try different keys for conversation messages
            for key in ["conversations", "messages", "dialogue", "turns"]:
                if key in conversation:
                    messages = conversation[key]
                    break

        if not messages:
            return []

        # Parse turns
        turns = []
        current_turn = {}

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            # Detect role/from field
            role = None
            content = None

            # OpenAI format: {"role": "user", "content": "..."}
            if "role" in msg and "content" in msg:
                role = msg["role"]
                content = msg["content"]
            # ShareGPT format: {"from": "human", "value": "..."}
            elif "from" in msg and "value" in msg:
                role = msg["from"]
                content = msg["value"]

            if not role or not content:
                continue

            # Normalize role
            role = role.lower()
            if role in ["user", "human", "question"]:
                if current_turn.get("user"):
                    # New turn starts, save previous turn
                    turns.append(current_turn)
                    current_turn = {}
                current_turn["user"] = content
            elif role in ["assistant", "gpt", "ai", "answer", "response"]:
                current_turn["assistant"] = content
                # Turn complete, save it
                if current_turn.get("user"):
                    turns.append(current_turn)
                    current_turn = {}

        return turns

    def _format_conversation_history(self, conversation_history: list[dict[str, str]]) -> str:
        """Format conversation history as input context.

        Args:
            conversation_history: List of turns

        Returns:
            Formatted history string
        """
        lines = []
        for turn in conversation_history:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")

        return "\n".join(lines)

    def load_and_merge_files(self, dataset_path: str, useful_files: list[str] | None = None) -> pd.DataFrame:
        """Load and merge all useful files in dataset directory.

        Args:
            dataset_path: Path to dataset directory
            useful_files: List of useful file paths (relative to dataset_path).
                         If None, load all supported files.

        Returns:
            Merged DataFrame
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        # Find files to load
        if useful_files:
            file_paths = [dataset_path / f for f in useful_files]
        else:
            # Auto-detect all supported files
            file_paths = self._find_data_files(dataset_path)

        if not file_paths:
            raise ValueError(f"No data files found in {dataset_path}")

        logger.info(f"Loading {len(file_paths)} files from {dataset_path}")

        # Load all files
        dataframes = []
        for file_path in file_paths:
            try:
                df = self._load_file(file_path)
                logger.info(f"Loaded {len(df)} rows from {file_path.name}")
                dataframes.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        if not dataframes:
            raise ValueError("Failed to load any files")

        # Merge all dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Merged {len(dataframes)} files into {len(merged_df)} total rows")

        return merged_df

    def _find_data_files(self, dataset_path: Path) -> list[Path]:
        """Find all supported data files in directory.

        Args:
            dataset_path: Dataset directory path

        Returns:
            List of file paths
        """
        supported_extensions = [".csv", ".json", ".jsonl", ".parquet", ".arrow"]
        file_paths = []

        for ext in supported_extensions:
            file_paths.extend(dataset_path.rglob(f"*{ext}"))

        # Filter out hidden files and cache
        file_paths = [
            f for f in file_paths if not any(part.startswith(".") for part in f.parts) and "__pycache__" not in str(f)
        ]

        return sorted(file_paths)

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """Load single data file.

        Args:
            file_path: File path

        Returns:
            DataFrame
        """
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(file_path)
        if suffix == ".json":
            return pd.read_json(file_path, lines=False)
        if suffix == ".jsonl":
            return pd.read_json(file_path, lines=True)
        if suffix == ".parquet":
            return pd.read_parquet(file_path)
        if suffix == ".arrow":
            # Read arrow file using pyarrow

            table = pq.read_table(file_path)
            return table.to_pandas()
        raise ValueError(f"Unsupported file format: {suffix}")

    def save_alpaca_json(self, alpaca_samples: list[dict[str, str]], output_path: str) -> None:
        """Save Alpaca-format samples to JSON file.

        Args:
            alpaca_samples: List of Alpaca samples
            output_path: Output JSON file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(alpaca_samples, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(alpaca_samples)} samples to {output_path}")
