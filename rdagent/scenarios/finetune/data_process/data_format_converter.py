"""
Data Format Converter for LLM Fine-tuning

Handles dataset format conversion to LLaMA-Factory compatible formats.
This module is specifically for format conversion, not general data processing.
"""

from pathlib import Path

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.finetune.tasks import create_llm_finetune_tasks
from rdagent.scenarios.finetune.train.coder import LLMPipelineCoSTEER
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow.misc import wait_retry


class DataFormatConverter:
    """Converts datasets to LLaMA-Factory compatible formats"""

    def __init__(self, dataset: str, model: str, ft_rd_setting, scen):
        self.dataset = dataset
        self.model = model
        self.ft_rd_setting = ft_rd_setting
        self.scen = scen
        self.coder = LLMPipelineCoSTEER(scen)

    def convert_dataset(self, env, shared_workspace_dir: Path) -> bool:
        """
        Convert dataset to LLaMA-Factory compatible format.

        Args:
            env: The execution environment
            shared_workspace_dir: Directory to store converted data

        Returns:
            bool: True if conversion succeeded, False otherwise
        """
        logger.info(f"Converting dataset format for {self.dataset}...")

        # Create data format conversion experiment
        conversion_exp = self._create_conversion_experiment(env)

        # Develop the experiment code
        conversion_exp = self.coder.develop(conversion_exp)

        # Execute the format conversion
        success = self._execute_conversion(conversion_exp, env)

        if success:
            logger.info("Data format conversion completed successfully")
            # Verify output files exist
            self._verify_converted_data(shared_workspace_dir)
        else:
            logger.error("Data format conversion failed")

        return success

    def _create_conversion_experiment(self, env) -> DSExperiment:
        """Create data format conversion experiment"""

        # Get runtime environment information
        runtime_info = get_runtime_environment_by_env(env)

        # Get dataset samples
        data_samples = self._get_dataset_samples()

        # Create data format conversion task
        task = create_llm_finetune_tasks(self.dataset, self.model)[0]  # First task is data format conversion

        # Set task description using template
        task.description = T("scenarios.finetune.data_process.prompts:data_format_task_prompt").r(
            dataset=self.dataset,
            runtime_info=runtime_info,
            data_samples=data_samples,
        )

        return DSExperiment(pending_tasks_list=[[task]])

    def _execute_conversion(self, exp: DSExperiment, env) -> bool:
        """Execute data format conversion experiment"""

        if not exp.is_ready_to_run():
            logger.error("Data format conversion experiment is not ready to run")
            return False

        workspace = exp.experiment_workspace
        if workspace and hasattr(workspace, "run"):
            result = workspace.run(env=env, entry="python main.py")
            logger.info(f"Data conversion execution result: {result.exit_code}")
            if result.stdout:
                logger.info(f"Data conversion output:\n{result.stdout}")
            return result.exit_code == 0
        else:
            logger.warning("No executable workspace found for data conversion")
            return False

    def _get_dataset_samples(self) -> str:
        """Get dataset samples for processing"""
        try:
            # In Docker environment, dataset is mounted at /workspace/llm_finetune/data/raw
            dataset_path = Path("/workspace/llm_finetune/data/raw") / self.dataset
            if dataset_path.exists():
                return f"Dataset path: {dataset_path}\nPlease load and analyze the dataset from this path."

            # Fallback to local path for non-Docker environments
            local_dataset_path = Path(self.ft_rd_setting.local_data_path) / self.dataset
            if local_dataset_path.exists():
                return f"Dataset path: {local_dataset_path}\nPlease load and analyze the dataset from this path."

            return f"Dataset {self.dataset} not found. Please ensure it's available at the expected path."
        except Exception as e:
            logger.warning(f"Could not load dataset samples: {e}")
            return f"Dataset: {self.dataset}\nPlease download and analyze the dataset."

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
