"""
Data Format Converter for LLM Fine-tuning

Handles dataset format conversion to LLaMA-Factory compatible formats.
Supports all LLaMA-Factory data formats with dynamic file discovery.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from rdagent.components.coder.finetune.data_format import (
    DataFormatCoSTEER,
    DataFormatTask,
)
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.log import rdagent_logger as logger


class DataFormatConverter:
    """Converts datasets to LLaMA-Factory compatible formats using CoSTEER architecture"""

    # Supported LLaMA-Factory data formats
    SUPPORTED_FORMATS = {".json", ".jsonl", ".csv", ".parquet", ".arrow"}
    CONFIG_FILE = "dataset_info.json"

    def __init__(self, dataset: str, scen):
        self.dataset = dataset
        self.coder = DataFormatCoSTEER(scen)

    def convert_dataset(self, env, preprocessed_dir: Path) -> bool:
        """Convert dataset to LLaMA-Factory compatible format using CoSTEER"""
        logger.info(f"Converting dataset format for {self.dataset}...")

        # Create task and experiment
        task = DataFormatTask(name=f"DataFormat_{self.dataset}", dataset=self.dataset)
        exp = Experiment([task])

        # Develop using CoSTEER
        exp = self.coder.develop(exp)

        # Execute the best implementation
        if exp.sub_workspace_list:
            workspace = exp.sub_workspace_list[0]
            success = self._execute_conversion(workspace, env)

            if success:
                logger.info("Data format conversion completed successfully")
                self._copy_converted_data(workspace, preprocessed_dir)
                self._verify_converted_data(preprocessed_dir)
            return success

        return False

    def _execute_conversion(self, workspace: FBWorkspace, env) -> bool:
        """Execute data format conversion code"""
        if workspace and hasattr(workspace, "run"):
            result = workspace.run(env=env, entry="python main.py")
            logger.info(f"Data conversion execution result: {result.exit_code}")
            if result.stdout:
                logger.info(f"Data conversion output:\n{result.stdout}")
            return result.exit_code == 0
        else:
            logger.error("No executable workspace found for data conversion")
            return False

    def _copy_converted_data(self, workspace: FBWorkspace, preprocessed_datasets: Path):
        """Copy the entire output directory to preprocessed_datasets"""

        workspace_output_dir = workspace.workspace_path / "output"

        # Check if output directory exists
        if not workspace_output_dir.exists():
            logger.warning(f"Output directory not found: {workspace_output_dir}")
            return

        # Ensure preprocessed_datasets directory exists
        preprocessed_datasets.mkdir(parents=True, exist_ok=True)

        logger.info(f"Source directory: {workspace_output_dir}")
        logger.info(f"Target directory: {preprocessed_datasets}")

        try:
            # Check if output directory has files
            output_items = list(workspace_output_dir.iterdir())
            logger.info(f"Found {len(output_items)} items in output directory: {[item.name for item in output_items]}")

            # Simple: copy everything from output to preprocessed_datasets
            for item in output_items:
                dst = preprocessed_datasets / item.name
                if item.is_file():
                    shutil.copy2(item, dst)
                    logger.info(f"Copied file: {item.name}")
                elif item.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(item, dst)
                    logger.info(f"Copied directory: {item.name}")

            logger.info(f"Successfully copied output directory to {preprocessed_datasets}")

        except Exception as e:
            logger.error(f"Failed to copy output directory: {e}")

    def _validate_data_file(self, file_path: Path) -> bool:
        """Validate a single data file based on its format"""
        try:
            suffix = file_path.suffix.lower()

            if suffix == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return isinstance(data, list) and len(data) > 0

            elif suffix == ".jsonl":
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if not lines:
                        return False
                    # Validate first line is valid JSON
                    json.loads(lines[0])
                return True

            elif suffix == ".csv":
                df = pd.read_csv(file_path)
                return len(df) > 0 and len(df.columns) > 0

            elif suffix == ".parquet":
                df = pd.read_parquet(file_path)
                return len(df) > 0 and len(df.columns) > 0

            elif suffix == ".arrow":
                df = pd.read_feather(file_path)
                return len(df) > 0 and len(df.columns) > 0

            return False

        except Exception as e:
            logger.warning(f"Failed to validate {file_path.name}: {e}")
            return False

    def _validate_config_file(self, file_path: Path) -> bool:
        """Validate dataset_info.json configuration file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Check basic structure
            if not isinstance(config, dict):
                return False

            # Each dataset entry should have required fields
            for dataset_name, dataset_config in config.items():
                if not isinstance(dataset_config, dict):
                    continue
                if "file_name" not in dataset_config:
                    logger.warning(f"Missing 'file_name' in dataset config: {dataset_name}")
                    return False

            return True

        except Exception as e:
            logger.warning(f"Failed to validate config file {file_path.name}: {e}")
            return False

    def _verify_converted_data(self, preprocessed_datasets: Path):
        """Verify all converted data with format-specific validation"""
        if not preprocessed_datasets.exists():
            logger.error(f"Preprocessed directory does not exist: {preprocessed_datasets}")
            return

        # Discover files in preprocessed directory
        all_files = list(preprocessed_datasets.iterdir())
        data_files = [f for f in all_files if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS]
        config_files = [f for f in all_files if f.is_file() and f.name == self.CONFIG_FILE]

        # Validate data files
        valid_data_files = []
        for data_file in data_files:
            if self._validate_data_file(data_file):
                valid_data_files.append(data_file.name)
                logger.info(f"✓ Validated data file: {data_file.name}")
            else:
                logger.error(f"✗ Invalid data file: {data_file.name}")

        # Validate configuration files
        valid_config_files = []
        for config_file in config_files:
            if self._validate_config_file(config_file):
                valid_config_files.append(config_file.name)
                logger.info(f"✓ Validated config file: {config_file.name}")
            else:
                logger.error(f"✗ Invalid config file: {config_file.name}")

        # Summary
        if not data_files:
            logger.error(f"No data files found in preprocessed directory: {preprocessed_datasets}")
        elif not valid_data_files:
            logger.error("No valid data files found")
        elif not config_files:
            logger.error(f"No {self.CONFIG_FILE} found")
        elif not valid_config_files:
            logger.error(f"Invalid {self.CONFIG_FILE}")
        else:
            logger.info(f"✓ Data conversion verification successful:")
            logger.info(f"  - {len(valid_data_files)} valid data files: {valid_data_files}")
            logger.info(f"  - {len(valid_config_files)} valid config files: {valid_config_files}")
