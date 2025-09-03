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

    def __init__(self, dataset: str, model: str, ft_rd_setting, scen):
        self.dataset = dataset
        self.model = model
        self.ft_rd_setting = ft_rd_setting
        self.scen = scen
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

    def _discover_generated_files(self, workspace_data_dir: Path) -> Tuple[List[Path], List[Path]]:
        """Discover all generated files and categorize them"""
        if not workspace_data_dir.exists():
            return [], []

        all_files = list(workspace_data_dir.iterdir())
        data_files = []
        config_files = []

        for file_path in all_files:
            if file_path.is_file():
                if file_path.name == self.CONFIG_FILE:
                    config_files.append(file_path)
                elif file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    data_files.append(file_path)

        return data_files, config_files

    def _copy_converted_data(self, workspace: FBWorkspace, preprocessed_dir: Path):
        """Copy all generated data files with dynamic discovery"""

        workspace_data_dir = workspace.workspace_path / "data"
        data_files, config_files = self._discover_generated_files(workspace_data_dir)

        # Ensure preprocessed directory exists
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

        # Copy all data files
        copied_files = []
        for src_file in data_files:
            dst_file = preprocessed_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            copied_files.append(src_file.name)
            logger.info(f"Copied data file: {src_file.name}")

        # Copy configuration files
        for src_file in config_files:
            dst_file = preprocessed_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            copied_files.append(src_file.name)
            logger.info(f"Copied config file: {src_file.name}")

        if not copied_files:
            logger.warning("No supported data files found in workspace/data/")
        else:
            logger.info(f"Successfully copied {len(copied_files)} files: {copied_files}")

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

    def _verify_converted_data(self, preprocessed_dir: Path):
        """Verify all converted data with format-specific validation"""
        if not preprocessed_dir.exists():
            logger.error(f"Preprocessed directory does not exist: {preprocessed_dir}")
            return

        # Discover files in preprocessed directory
        all_files = list(preprocessed_dir.iterdir())
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
            logger.error("No data files found in preprocessed directory")
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
