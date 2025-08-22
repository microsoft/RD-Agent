"""
Data Format Converter for LLM Fine-tuning

Handles dataset format conversion to LLaMA-Factory compatible formats.
This module is specifically for format conversion, not general data processing.
"""

from pathlib import Path
from typing import Optional

from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger


class DataFormatConverter:
    """Converts datasets to LLaMA-Factory compatible formats using CoSTEER architecture"""

    def __init__(self, dataset: str, model: str, ft_rd_setting, scen):
        self.dataset = dataset
        self.model = model
        self.ft_rd_setting = ft_rd_setting
        self.scen = scen

        from rdagent.components.coder.finetune.data_format import DataFormatCoSTEER

        self.coder = DataFormatCoSTEER(scen)

    def convert_dataset(self, env, shared_workspace_dir: Path) -> bool:
        """Convert dataset to LLaMA-Factory compatible format using CoSTEER"""
        logger.info(f"Converting dataset format for {self.dataset}...")

        from rdagent.components.coder.finetune.data_format import DataFormatTask
        from rdagent.core.experiment import Experiment

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
                self._verify_converted_data(shared_workspace_dir)
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

    def get_dataset_samples(self) -> str:
        """Get dataset samples for processing"""
        import json

        import pandas as pd

        try:
            # Try Docker environment path first
            dataset_path = Path("/workspace/llm_finetune/data/raw") / self.dataset
            if not dataset_path.exists():
                # Fallback to local path for non-Docker environments
                dataset_path = Path(self.ft_rd_setting.local_data_path) / self.dataset

            if not dataset_path.exists():
                return f"Dataset {self.dataset} not found at expected paths."

            # Find data files in the dataset directory
            data_files = []
            for ext in ["*.json", "*.jsonl", "*.csv", "*.parquet"]:
                data_files.extend(list(dataset_path.glob(ext)))

            if not data_files:
                return f"No supported data files found in {dataset_path}. Supported formats: json, jsonl, csv, parquet"

            # Load sample from the first data file
            data_file = data_files[0]
            sample_data = None

            if data_file.suffix.lower() == ".json":
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        sample_data = data[0]
                    elif isinstance(data, dict):
                        sample_data = data

            elif data_file.suffix.lower() == ".jsonl":
                with open(data_file, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        sample_data = json.loads(first_line)

            elif data_file.suffix.lower() == ".csv":
                df = pd.read_csv(data_file)
                if len(df) > 0:
                    sample_data = df.iloc[0].to_dict()

            elif data_file.suffix.lower() == ".parquet":
                df = pd.read_parquet(data_file)
                if len(df) > 0:
                    sample_data = df.iloc[0].to_dict()

            if sample_data is not None:
                return json.dumps(sample_data, ensure_ascii=False, indent=2)
            else:
                return f"Could not extract sample from {data_file.name}"

        except Exception as e:
            logger.warning(f"Could not load dataset samples: {e}")
            return f"Dataset: {self.dataset}\nError loading samples: {str(e)}"

    def _verify_converted_data(self, shared_workspace_dir: Path):
        """Verify that converted data exists in the shared workspace"""
        # Use the actual shared workspace directory (host path) instead of Docker container path
        processed_data_path = shared_workspace_dir

        # Check for expected output files from LLaMA-Factory format conversion
        expected_files = ["processed_dataset.json", "dataset_info.json"]
        missing_files = []

        for file_name in expected_files:
            file_path = processed_data_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)

        if missing_files:
            logger.warning(f"Expected converted files not found: {missing_files} in {processed_data_path}")
            logger.warning("Please ensure the conversion script generates these files")
        else:
            logger.info(f"All expected converted data files found in {processed_data_path}")
