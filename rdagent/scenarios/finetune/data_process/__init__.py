"""
Data processing module for LLM fine-tuning

This module contains utilities and components for processing data
in the LLM fine-tuning pipeline.

Current components:
- data_format_converter: Convert datasets to LLaMA-Factory compatible formats

Future components may include:
- data_cleaning: Clean and preprocess raw data
- data_augmentation: Augment training data
- data_validation: Validate data quality
- data_sampling: Sample data for training/evaluation
"""

from pathlib import Path

from rdagent.scenarios.finetune.data_process.data_format_converter import (
    DataFormatConverter,
)

# Module version
__version__ = "0.1.0"

# Module root path
MODULE_ROOT = Path(__file__).parent

# Export main classes for easier imports
__all__ = ["DataFormatConverter"]
