"""
Data components for RD-Agent.

This module provides automated dataset search, download, and SFT preparation capabilities.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from rdagent.components.data.data_cleaner import DataCleaner
from rdagent.components.data.data_converter import DataConverter
from rdagent.components.data.dataset_agent import DatasetSearchAgent
from rdagent.components.data.dataset_inspector import DatasetInspector
from rdagent.components.data.dataset_manager import DatasetManager
from rdagent.components.data.schema_analyzer import SchemaAnalyzer
from rdagent.components.data.sft_processor import SFTProcessingSettings, SFTProcessor


def convert_to_sft(input_path: str, output_file: str, task_description: str = "") -> Dict[str, Any]:
    """
    Fully autonomous SFT dataset converter with intelligent routing.

    Automatically selects the best processing path based on data quality:
    - High quality data → Light path (simple conversion + parallel quality scoring)
    - Low quality/complex data → Heavy path (parallel LLM deep conversion)

    Args:
        input_path: Path to input dataset (file or directory)
        output_file: Path to output Alpaca JSON file
        task_description: Description of the task/dataset (helps LLM understand context)

    Returns:
        Processing result dict with stats and output path
        {
            "success": bool,
            "output_path": str,
            "stats": {...},
            "processing_path": "light" | "heavy"
        }

    Example:
        >>> result = convert_to_sft(
        ...     input_path="data/qa_dataset.csv",
        ...     output_file="output/alpaca_data.json",
        ...     task_description="Medical Q&A dataset"
        ... )
        >>> print(f"Processed {result['stats']['successful_rows']} samples")
    """
    from rdagent.oai.llm_utils import APIBackend

    # Create settings with defaults (fully autonomous)
    settings = SFTProcessingSettings(
        input_path=Path(input_path), output_file=Path(output_file), task_description=task_description
    )

    # Create processor with default LLM
    llm_client = APIBackend()
    processor = SFTProcessor(settings, llm_client)

    # Process dataset (fully autonomous)
    result = processor.process()

    return result


__all__ = [
    "DatasetSearchAgent",
    "DatasetInspector",
    "DatasetManager",
    "SchemaAnalyzer",
    "DataConverter",
    "DataCleaner",
    "SFTProcessor",
    "SFTProcessingSettings",
    "convert_to_sft",
]
