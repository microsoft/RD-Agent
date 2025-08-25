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
                self._copy_converted_data(workspace, shared_workspace_dir)
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

    def _copy_converted_data(self, workspace: "FBWorkspace", shared_workspace_dir: Path):
        """Copy converted data files from workspace/data to shared directory"""
        import shutil

        expected_files = ["processed_dataset.json", "dataset_info.json"]

        for file_name in expected_files:
            # Look for files in workspace/data/ subdirectory
            src_file = workspace.workspace_path / "data" / file_name
            dst_file = shared_workspace_dir / file_name

            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                logger.info(f"Copied {file_name} to shared workspace")
            else:
                logger.warning(f"Expected file {file_name} not found in workspace/data/")

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
