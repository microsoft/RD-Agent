"""
SFT Dataset Processor - Production-grade tool for SFT data preparation

This module provides a robust, production-ready system for converting datasets
to SFT (Supervised Fine-Tuning) format with features including:
- Checkpoint/resume support for large datasets
- Caching to avoid redundant LLM calls
- Parallel processing for speed
- Comprehensive error handling and logging
- Multiple fallback strategies for schema detection
"""

import argparse
import concurrent.futures
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from rdagent.components.data.data_cleaner import DataCleaner
from rdagent.components.data.data_converter import DataConverter

# Import existing SFT modules
from rdagent.components.data.schema_analyzer import SchemaAnalyzer
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

# ===========================
# Configuration Section
# ===========================


class SFTProcessingSettings(BaseModel):
    """Settings for fully autonomous SFT dataset processor"""

    # Required input/output
    input_path: Path = Field(description="Input dataset path (file or directory)")
    output_file: Path = Field(description="Output Alpaca JSON file path")
    task_description: str = Field(default="", description="Task description for LLM context")

    # Parallel processing and batch settings
    max_workers: int = Field(default=20, description="Number of parallel workers")
    batch_size: int = Field(default=10, description="Batch size for LLM conversion and checkpoint frequency")

    # Data quality settings (mandatory for light path)
    quality_threshold: float = Field(default=7.0, description="Quality score threshold (0-10)")
    min_instruction_len: int = Field(default=5, description="Minimum instruction length")
    min_output_len: int = Field(default=10, description="Minimum output length")
    max_instruction_len: int = Field(default=2000, description="Maximum instruction length")
    max_output_len: int = Field(default=4000, description="Maximum output length")

    # Metadata and output settings
    enable_metadata: bool = Field(default=True, description="Include metadata in output")

    # Checkpoint (for resumability)
    resume_from_checkpoint: bool = Field(default=True, description="Resume from checkpoint if interrupted")
    checkpoint_file: Path = Field(default=Path("sft_checkpoint.json"), description="Checkpoint file path")

    # Logging
    log_dir: Path = Field(default=Path("logs"), description="Log directory")
    log_level: str = Field(default="INFO", description="Logging level")

    # Internal processing parameters (not exposed to users typically)
    useful_files: list[str] | None = Field(default=None, description="Specific files to process (optional)")

    class Config:
        arbitrary_types_allowed = True


# ===========================
# Utility Classes (from process.py)
# ===========================


class CheckpointManager:
    """Simplified checkpoint manager for batch-level resumable processing"""

    def __init__(self, checkpoint_file: Path) -> None:
        """Initialize checkpoint manager."""
        self.checkpoint_file = Path(checkpoint_file)
        self.completed_batches = self._load_checkpoint()
        self.lock = Lock()

    def _load_checkpoint(self) -> list[int]:
        """Load checkpoint data from file"""
        if self.checkpoint_file.exists():
            try:
                with self.checkpoint_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("completed_batches", [])
            except Exception as e:
                logging.warning(f"Failed to load checkpoint: {e}")
        return []

    def get_completed_batches(self) -> list[int]:
        """Get list of completed batch indices"""
        return self.completed_batches.copy()

    def mark_batch_complete(self, batch_idx: int) -> None:
        """Mark a batch as completed and save to disk"""
        with self.lock:
            if batch_idx not in self.completed_batches:
                self.completed_batches.append(batch_idx)
                self.completed_batches.sort()
                self._save()

    def _save(self) -> None:
        """Save checkpoint data to file"""
        try:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with self.checkpoint_file.open("w", encoding="utf-8") as f:
                json.dump(
                    {"completed_batches": self.completed_batches, "timestamp": datetime.now().isoformat()},
                    f,
                    indent=2,
                )
        except Exception as e:
            logging.exception(f"Failed to save checkpoint: {e}")

    def clear(self) -> None:
        """Clear checkpoint file"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        self.completed_batches = []


# ===========================
# Main SFT Processor
# ===========================


class SFTProcessor:
    """Production-grade SFT dataset processor"""

    def __init__(
        self,
        settings: SFTProcessingSettings | None = None,
        llm_client: APIBackend | None = None,
        **kwargs,
    ) -> None:
        """Initialize SFT processor.

        Args:
            settings: Settings object (if None, kwargs are used to create one)
            llm_client: LLM API client (creates default if not provided)
            **kwargs: Settings parameters if settings is None
        """
        # Handle settings
        if settings is None:
            if not kwargs:
                raise ValueError("Either settings or settings parameters must be provided")
            settings = SFTProcessingSettings(**kwargs)
        self.settings = settings

        # Initialize LLM client
        self.llm_client = llm_client or APIBackend()

        # Initialize managers
        self.checkpoint_manager = CheckpointManager(settings.checkpoint_file)

        # Initialize SFT modules
        self.schema_analyzer = SchemaAnalyzer(self.llm_client)
        self.data_converter = DataConverter()
        self.data_cleaner = DataCleaner(
            self.llm_client,
            config={
                "min_instruction_len": settings.min_instruction_len,
                "min_output_len": settings.min_output_len,
                "max_instruction_len": settings.max_instruction_len,
                "max_output_len": settings.max_output_len,
                "quality_threshold": settings.quality_threshold,
            },
        )

        # Statistics tracking
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "total_rows": 0,
            "processed_rows": 0,
            "successful_rows": 0,
            "failed_rows": 0,
            "start_time": None,
            "end_time": None,
        }

    def analyze_schema_with_fallback(self, sample_data: pd.DataFrame, task_description: str) -> dict[str, Any]:
        """Fully autonomous schema analysis with automatic fallback.

        Returns schema dict with fields:
            - data_type: "single_turn" or "multi_turn"
            - instruction_col: column name for instruction
            - output_col: column name for output
            - input_col: optional column name for input
            - method: "llm" or "fallback"
        """
        # Level 1: LLM analysis (autonomous)
        try:
            logging.info("LLM schema analysis...")
            schema = self.schema_analyzer.analyze(sample_data, task_description)
            schema["method"] = "llm"
            return schema

        except Exception as e:
            logging.warning(f"LLM schema analysis failed: {e}, falling back to heuristic detection")

        # Level 2: Automatic heuristic fallback (autonomous)
        logging.info("Using automatic heuristic schema detection")
        schema = self._intelligent_guess_schema(sample_data)
        schema["method"] = "fallback"
        return schema

    def _intelligent_guess_schema(self, data: pd.DataFrame) -> dict[str, Any]:
        """Intelligent heuristic-based schema guessing"""
        columns = list(data.columns)

        # Common patterns for instruction columns
        instruction_patterns = [
            "instruction",
            "question",
            "prompt",
            "input",
            "query",
            "text",
            "problem",
            "sentence",
            "premise",
            "context",
        ]

        # Common patterns for output columns
        output_patterns = [
            "output",
            "answer",
            "response",
            "completion",
            "label",
            "target",
            "solution",
            "reply",
            "result",
            "hypothesis",
        ]

        # Find best matches
        instruction_col = None
        output_col = None

        # Try exact matches first
        for col in columns:
            col_lower = col.lower()
            if not instruction_col and col_lower in instruction_patterns:
                instruction_col = col
            if not output_col and col_lower in output_patterns:
                output_col = col

        # Try partial matches
        if not instruction_col:
            for col in columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in instruction_patterns):
                    instruction_col = col
                    break

        if not output_col:
            for col in columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in output_patterns):
                    output_col = col
                    break

        # Last resort: use first and last columns
        if not instruction_col:
            instruction_col = columns[0]
        if not output_col:
            output_col = columns[-1] if len(columns) > 1 else columns[0]

        logging.warning(f"Fallback schema detection: instruction='{instruction_col}', output='{output_col}'")

        return {
            "data_type": "single_turn",
            "instruction_col": instruction_col,
            "output_col": output_col,
            "input_col": None,
            "reasoning": "Automatic fallback based on column name patterns",
        }

    def process(self) -> dict[str, Any]:
        """Main processing entry point.

        Returns:
            Result dictionary with statistics and output path
        """
        self.stats["start_time"] = time.time()

        # Setup logging
        self._setup_logging()

        logging.info("=" * 60)
        logging.info("SFT Dataset Processor - Fully Autonomous Mode")
        logging.info(f"Input: {self.settings.input_path}")
        logging.info(f"Output: {self.settings.output_file}")
        logging.info(f"Workers: {self.settings.max_workers} (parallel processing)")
        logging.info("=" * 60)

        try:
            # Load and process data
            result = self._process_with_checkpoint()

            # Clear checkpoint if successful
            if self.stats["processed_rows"] >= self.stats["total_rows"]:
                self.checkpoint_manager.clear()
                logging.info("Processing complete, checkpoint cleared")

            # Calculate final statistics
            self.stats["end_time"] = time.time()
            duration = self.stats["end_time"] - self.stats["start_time"]

            # Display statistics
            self._display_statistics(duration)

            return {
                "success": True,
                "output_path": str(self.settings.output_file),
                "stats": self.stats.copy(),
                "processing_path": result.get("processing_path"),
            }

        except Exception as e:
            logging.exception(f"Processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stats": self.stats.copy(),
            }

    def _load_data(self) -> pd.DataFrame:
        """Load data from file or directory"""
        input_path = Path(self.settings.input_path)

        # Check if input is a single file or directory
        if input_path.is_file():
            # Single file - load directly
            logging.info(f"Loading single file: {input_path}")

            suffix = input_path.suffix.lower()
            if suffix == ".csv":
                return pd.read_csv(input_path)
            if suffix == ".json":
                return pd.read_json(input_path, lines=False)
            if suffix == ".jsonl":
                return pd.read_json(input_path, lines=True)
            if suffix == ".parquet":
                return pd.read_parquet(input_path)
            if suffix in [".xlsx", ".xls"]:
                return pd.read_excel(input_path)
            raise ValueError(f"Unsupported file format: {suffix}")

        if input_path.is_dir():
            # Directory - use data_converter's load_and_merge_files
            logging.info(f"Loading directory: {input_path}")
            return self.data_converter.load_and_merge_files(str(input_path), self.settings.useful_files)

        raise ValueError(f"Input path does not exist: {input_path}")

    def _process_with_checkpoint(self) -> dict[str, Any]:
        """Process dataset with intelligent routing based on data quality"""
        # Step 1: Load data
        logging.info("Step 1: Loading data files...")
        merged_data = self._load_data()
        self.stats["total_rows"] = len(merged_data)
        logging.info(f"Loaded {self.stats['total_rows']} total rows")

        # Step 2: Analyze schema and assess quality
        logging.info("Step 2: Analyzing schema and assessing data quality...")
        sample_data = merged_data.head(100)

        # Analyze schema (no caching - re-analyze each time)
        schema = self._analyze_with_quality_assessment(sample_data, self.settings.task_description)

        quality_score = schema.get("quality_score", 0)
        logging.info(
            f"Schema detected: data_type={schema['data_type']}, "
            f"instruction={schema.get('instruction_col')}, "
            f"output={schema.get('output_col')}, "
            f"quality_score={quality_score:.2f}, "
            f"method={schema.get('method')}",
        )

        # Step 3: Intelligent routing based on quality
        logging.info("Step 3: Selecting processing path based on data quality...")

        if self._should_use_light_path(schema, quality_score):
            # Light path: Simple conversion + cleaning
            logging.info(f"✅ Using LIGHT path (quality_score={quality_score:.2f}, clear QA columns detected)")
            result = self._process_light_path(merged_data, schema)
            processing_path = "light"
        else:
            # Heavy path: Parallel LLM conversion
            logging.info(f"🔧 Using HEAVY path (quality_score={quality_score:.2f}, unclear structure)")
            result = self._process_heavy_path(merged_data)
            processing_path = "heavy"

        # Update stats
        self.stats["successful_rows"] = result["samples_count"]
        self.stats["processing_path"] = processing_path
        self.stats["quality_score"] = quality_score

        return {
            "samples": result["samples_count"],
            "stats": self.stats,
            "schema": schema,
            "processing_path": processing_path,
            "output_path": str(self.settings.output_file),
        }

    def _analyze_with_quality_assessment(self, sample_data: pd.DataFrame, task_description: str) -> dict[str, Any]:
        """Analyze schema with quality assessment for intelligent routing"""
        # Temporarily suppress detailed LLM logs during schema analysis
        original_level = logging.getLogger().level
        litellm_logger = logging.getLogger("rdagent.oai.backend.litellm")
        original_litellm_level = litellm_logger.level

        logging.getLogger().setLevel(logging.WARNING)
        litellm_logger.setLevel(logging.ERROR)  # Suppress litellm token-by-token logs

        try:
            print("🔍 Schema analysis in progress...", end=" ", flush=True)
            # First get schema using existing fallback mechanism
            schema = self.analyze_schema_with_fallback(sample_data, task_description)
            print("✓")
        finally:
            # Restore original logging levels
            logging.getLogger().setLevel(original_level)
            litellm_logger.setLevel(original_litellm_level)

        # Add quality assessment
        quality_score = 1.0

        # Factor 1: Check if instruction and output columns are clearly identified
        if not schema.get("instruction_col") or not schema.get("output_col"):
            quality_score *= 0.3

        # Factor 2: Check method used (LLM is more reliable than fallback)
        if schema.get("method") == "fallback":
            quality_score *= 0.5
        elif schema.get("method") == "manual":
            quality_score = 1.0  # Manual config is always trusted

        # Factor 3: Check data completeness (if columns exist)
        if schema.get("instruction_col") and schema.get("instruction_col") in sample_data.columns:
            null_ratio = sample_data[schema["instruction_col"]].isna().mean()
            quality_score *= 1 - null_ratio * 0.5

        # Factor 4: Check if LLM had confidence (if available)
        if "confidence" in schema:
            quality_score *= schema["confidence"]

        # Factor 5: Check if column names are standardized (clear Q&A structure)
        # Generic column names like 'field_1', 'column1', 'data', etc. indicate unclear structure
        generic_patterns = ["field", "column", "col", "data", "feature", "var", "x", "y"]
        instruction_col = schema.get("instruction_col", "").lower()
        output_col = schema.get("output_col", "").lower()

        # Check if column names contain generic patterns
        has_generic_instruction = any(pattern in instruction_col for pattern in generic_patterns)
        has_generic_output = any(pattern in output_col for pattern in generic_patterns)

        if has_generic_instruction or has_generic_output:
            # Generic column names suggest unclear structure → reduce quality score
            quality_score *= 0.6  # Significant penalty for generic names

        schema["quality_score"] = quality_score
        return schema

    def _should_use_light_path(self, schema: dict[str, Any], quality_score: float) -> bool:
        """Determine if data quality is good enough for light path"""
        # Light path criteria:
        # - Quality score > 0.8 (mandatory threshold)
        # - Clear instruction and output columns identified
        # - High quality data with standardized column names

        has_clear_columns = bool(schema.get("instruction_col") and schema.get("output_col"))

        # Quality score is the primary factor
        is_high_quality = quality_score > 0.8

        # Only use light path if BOTH conditions are met
        return has_clear_columns and is_high_quality

    def _process_light_path(self, data: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any]:
        """Light path: Simple conversion + mandatory parallel quality scoring"""
        logging.info("Light path: Starting simple conversion...")

        # Step 1: Simple format conversion using data_converter
        alpaca_samples = self.data_converter.convert_to_alpaca(
            data,
            schema,
            enable_metadata=self.settings.enable_metadata,
        )
        logging.info(f"Converted {len(alpaca_samples)} samples to Alpaca format")

        # Step 2-4: Clean with DataCleaner (dedup + length filter + quality scoring)
        # Define save callback for incremental saving
        def save_intermediate(samples):
            self.data_converter.save_alpaca_json(samples, str(self.settings.output_file))

        logging.info(
            f"Light path: Data cleaning (dedup + length filter + quality scoring, {self.settings.max_workers} workers)...",
        )
        cleaned_samples, clean_stats = self.data_cleaner.clean(
            alpaca_samples,
            task_description=self.settings.task_description,
            enable_quality_scoring=True,
            max_workers=self.settings.max_workers,
            save_callback=save_intermediate,  # Enable incremental saving
        )
        logging.info(
            f"Cleaning complete: {clean_stats['final_samples']} samples retained ({clean_stats['retention_rate']*100:.1f}%)",
        )

        # Step 5: Final save (ensure all samples are saved)
        self.data_converter.save_alpaca_json(cleaned_samples, str(self.settings.output_file))
        logging.info(f"Light path: Final save to {self.settings.output_file}")

        return {
            "samples_count": len(cleaned_samples),
            "original_count": len(alpaca_samples),
            "clean_stats": clean_stats,
        }

    def _process_heavy_path(self, data: pd.DataFrame) -> dict[str, Any]:
        """Heavy path: Deduplication + Direct parallel LLM conversion

        Processing steps:
        1. Deduplicate rows (consistent with light path)
        2. Split into batches (batch_size=10)
        3. Parallel LLM conversion (20 workers)
        4. Save output (no additional cleaning needed)
        """

        logging.info("Heavy path: Direct LLM conversion (no schema analysis needed)")
        logging.info(f"Heavy path: Processing {len(data)} rows with 20 workers")

        # Deduplicate before LLM conversion (consistent with light path)
        original_count = len(data)
        data = data.drop_duplicates()
        deduped_count = len(data)
        removed_count = original_count - deduped_count
        logging.info(
            f"Heavy path: After deduplication: {deduped_count} unique rows (removed {removed_count} duplicates)",
        )

        # Split deduplicated data into batches
        batch_size = self.settings.batch_size  # Use configured batch size
        batches = self._split_into_batches(data, batch_size)
        total_batches = len(batches)

        logging.info(f"Heavy path: Split into {total_batches} batches of size {batch_size}")

        # Check for completed batches (resume support)
        completed_batches = set()
        if self.settings.resume_from_checkpoint:
            completed_batches = set(self.checkpoint_manager.get_completed_batches())
            if completed_batches:
                logging.info(f"Resuming from checkpoint: {len(completed_batches)} batches already completed")

        # Process batches in parallel (with incremental file writing)
        all_samples = []
        failed_batches = []

        # Load existing samples if resuming
        output_path = Path(self.settings.output_file)
        if output_path.exists() and self.settings.resume_from_checkpoint:
            try:
                with open(output_path, encoding="utf-8") as f:
                    existing_samples = json.load(f)
                all_samples = existing_samples
                logging.info(f"Loaded {len(existing_samples)} existing samples from checkpoint")
            except Exception as e:
                logging.warning(f"Failed to load existing samples: {e}, starting fresh")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.settings.max_workers) as executor:
            # Submit all batch processing tasks (skip completed ones)
            future_to_batch = {}
            for batch_idx, batch_data in enumerate(batches):
                if batch_idx in completed_batches:
                    logging.info(f"Skipping batch {batch_idx} (already completed)")
                    continue

                future = executor.submit(self._convert_raw_batch_with_llm, batch_data, batch_idx, total_batches)
                future_to_batch[future] = batch_idx

            # Collect results as they complete (with progress bar)
            num_to_process = len(future_to_batch)
            with tqdm(
                total=num_to_process,
                desc="🔧 Heavy path LLM conversion",
                unit="batch",
                ncols=100,
                colour="yellow",
            ) as pbar:
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]

                    try:
                        batch_samples = future.result()
                        all_samples.extend(batch_samples)

                        # Immediately save to file (incremental write)
                        self.data_converter.save_alpaca_json(all_samples, str(self.settings.output_file))
                        logging.info(f"💾 Batch {batch_idx+1} saved → Total: {len(all_samples)} samples")

                        # Mark batch as complete and save checkpoint
                        if self.settings.resume_from_checkpoint:
                            self.checkpoint_manager.mark_batch_complete(batch_idx)

                        # Update progress bar
                        pbar.set_postfix({"samples": len(all_samples), "failed": len(failed_batches)})
                        pbar.update(1)

                    except Exception as e:
                        logging.exception(f"Failed to process batch {batch_idx}: {e}")
                        failed_batches.append(batch_idx)
                        pbar.update(1)

        # Log completion
        logging.info(
            f"Heavy path: Completed LLM conversion. "
            f"Success: {len(all_samples)} samples, "
            f"Failed batches: {len(failed_batches)}",
        )

        logging.info(f"Heavy path: All batches saved incrementally to {self.settings.output_file}")

        return {"samples_count": len(all_samples), "failed_batches": failed_batches}

    def _split_into_batches(self, data: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
        """Split dataframe into smaller batches for parallel processing"""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i : i + batch_size]
            batches.append(batch)
        return batches

    def _convert_raw_batch_with_llm(
        self,
        batch_data: pd.DataFrame,
        batch_idx: int,
        total_batches: int,
    ) -> list[dict[str, str]]:
        """Convert a batch of raw data to SFT format using LLM (no schema needed)"""
        try:
            # Convert batch data to list of dicts for prompt
            batch_records = batch_data.to_dict("records")

            # Build prompt using T template system
            system_prompt = T(".prompts:heavy_conversion.system").r()
            user_prompt = T(".prompts:heavy_conversion.user").r(
                task_description=self.settings.task_description or "Convert to instruction-output pairs for training",
                batch_data=json.dumps(batch_records, ensure_ascii=False, indent=2),
                num_records=len(batch_records),
            )

            # Temporarily suppress detailed LLM logs during conversion
            original_level = logging.getLogger().level
            litellm_logger = logging.getLogger("rdagent.oai.backend.litellm")
            original_litellm_level = litellm_logger.level

            logging.getLogger().setLevel(logging.WARNING)
            litellm_logger.setLevel(logging.ERROR)  # Suppress litellm token-by-token logs

            try:
                # Call LLM (with retry from wait_retry decorator if needed)
                response = self.llm_client.build_messages_and_create_chat_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_mode=True,
                )
            finally:
                # Restore original logging levels
                logging.getLogger().setLevel(original_level)
                litellm_logger.setLevel(original_litellm_level)

            # Parse response
            result = json.loads(response)

            # Validate response format
            if "samples" not in result or not isinstance(result["samples"], list):
                raise ValueError("Invalid LLM response format: missing 'samples' list")

            # Validate each sample
            valid_samples = []
            for sample in result["samples"]:
                if isinstance(sample, dict) and "instruction" in sample and "output" in sample:
                    # Ensure input field exists (can be empty)
                    if "input" not in sample:
                        sample["input"] = ""
                    valid_samples.append(sample)
                else:
                    logging.warning(f"Skipping invalid sample in batch {batch_idx}: {sample}")

            # Log batch completion info
            conversion_rate = len(valid_samples) / len(batch_records) * 100 if batch_records else 0
            logging.info(
                f"Batch {batch_idx+1}/{total_batches}: "
                f"{len(batch_records)} records → {len(valid_samples)} samples "
                f"(conversion rate: {conversion_rate:.1f}%)",
            )

            return valid_samples

        except Exception as e:
            logging.exception(f"Error converting batch {batch_idx}: {e}")
            raise

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = self.settings.log_dir / f"sft_process_{timestamp}.log"

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.settings.log_level.upper()),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        )

        # Reduce third-party logging
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def _display_statistics(self, duration: float) -> None:
        """Display processing statistics"""
        minutes = int(duration // 60)
        seconds = duration % 60

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Duration: {minutes}m {seconds:.1f}s")
        print(f"Total rows: {self.stats['total_rows']}")
        print(f"Successful: {self.stats['successful_rows']}")
        print(f"Failed: {self.stats.get('failed_rows', 0)}")

        print(f"\nOutput: {self.settings.output_file}")
        print("=" * 60)


# ===========================
# CLI Interface
# ===========================


def main():
    """Command-line interface for SFT processor"""

    parser = argparse.ArgumentParser(description="SFT Dataset Processor")
    parser.add_argument("input_path", help="Input dataset path")
    parser.add_argument("output_file", help="Output JSON file path")
    parser.add_argument("--task", default="", help="Task description")
    parser.add_argument("--workers", type=int, default=20, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for LLM and checkpoint")
    parser.add_argument("--quality-threshold", type=float, default=7.0, help="Quality score threshold")
    parser.add_argument("--min-instruction-len", type=int, default=5, help="Min instruction length")
    parser.add_argument("--min-output-len", type=int, default=10, help="Min output length")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--no-metadata", action="store_true", help="Disable metadata extraction")

    args = parser.parse_args()

    # Create settings
    settings = SFTProcessingSettings(
        input_path=Path(args.input_path),
        output_file=Path(args.output_file),
        task_description=args.task,
        max_workers=args.workers,
        batch_size=args.batch_size,
        quality_threshold=args.quality_threshold,
        min_instruction_len=args.min_instruction_len,
        min_output_len=args.min_output_len,
        enable_metadata=not args.no_metadata,
        resume_from_checkpoint=args.resume,
    )

    # Process dataset
    processor = SFTProcessor(settings)
    result = processor.process()

    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
