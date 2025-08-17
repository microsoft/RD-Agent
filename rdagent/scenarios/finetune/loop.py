"""
LLM Fine-tuning Loop Implementation

Simplified LLM fine-tuning loop with two main steps:
1. Data format conversion
2. Model fine-tuning
"""

from pathlib import Path

from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.finetune.tasks import create_llm_finetune_tasks

# Import LLM-specific components
from rdagent.scenarios.finetune.train.coder import LLMPipelineCoSTEER
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T


class LLMFinetuneRDLoop:
    """LLM fine-tuning loop with two steps: data format conversion -> fine-tuning"""

    def __init__(self, dataset: str, model: str, ft_rd_setting):
        self.dataset = dataset
        self.model = model
        self.ft_rd_setting = ft_rd_setting

        # Create scenario
        scen_class = import_class(ft_rd_setting.scen)
        self.scen = scen_class()

        # Create code generator
        self.coder = LLMPipelineCoSTEER(self.scen)

        # Create environment with volume mapping for data visibility
        data_volumes = {}
        if ft_rd_setting.local_data_path:
            # Input data should be read-only to protect original data
            data_volumes[ft_rd_setting.local_data_path] = {
                "bind": "/workspace/llm_finetune/data/raw",
                "mode": "ro",
            }

        # Get finetune base directory from config
        finetune_base_dir = Path(ft_rd_setting.file_path)
        finetune_base_dir.mkdir(parents=True, exist_ok=True)

        # Ensure output directory is visible outside container and writable
        output_dir = finetune_base_dir / "llm_finetune_output"
        # Clean output directory for fresh start
        import shutil

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data_volumes[str(output_dir)] = {
            "bind": "/workspace/llm_finetune/output",
            "mode": "rw",
        }

        # Create shared workspace directory to persist data between steps
        self.shared_workspace_dir = finetune_base_dir / "llm_finetune_shared_workspace"
        # Clean shared workspace for fresh start
        if self.shared_workspace_dir.exists():
            shutil.rmtree(self.shared_workspace_dir)
        self.shared_workspace_dir.mkdir(parents=True, exist_ok=True)
        data_volumes[str(self.shared_workspace_dir)] = {
            "bind": "/workspace/llm_finetune/shared",
            "mode": "rw",
        }

        # Import get_ft_env here to avoid circular imports
        from rdagent.app.finetune.llm.conf import get_ft_env

        self.env = get_ft_env(
            extra_volumes=data_volumes,
            running_timeout_period=None,  # No time limit
            enable_cache=False,
        )

        logger.info(f"Initialized LLM finetune loop for {model} on {dataset}")
        logger.info(f"Shared workspace: {self.shared_workspace_dir}")

    def run(self):
        """Run LLM fine-tuning pipeline"""
        logger.info("Starting LLM fine-tuning pipeline...")

        # Step 1: Data format conversion
        logger.info("Step 1: Converting dataset format...")
        data_exp = self._create_data_format_experiment()
        data_exp = self.coder.develop(data_exp)
        self._execute_experiment(data_exp, "Data Format Conversion")

        # Step 2: Model fine-tuning
        logger.info("Step 2: Fine-tuning model...")
        finetune_exp = self._create_finetuning_experiment()
        finetune_exp = self.coder.develop(finetune_exp)
        self._execute_experiment(finetune_exp, "Model Fine-tuning")

        logger.info("LLM fine-tuning pipeline completed!")

    def _create_data_format_experiment(self) -> DSExperiment:
        """Create data format conversion experiment"""

        # Get runtime environment information
        runtime_info = get_runtime_environment_by_env(self.env)

        # Get dataset samples (simplified implementation)
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

    def _create_finetuning_experiment(self) -> DSExperiment:
        """Create fine-tuning experiment"""

        # Get runtime environment information
        runtime_info = get_runtime_environment_by_env(self.env)

        # Get LLaMA-Factory usage guide
        llamafactory_guide = T("scenarios.finetune.train.prompts:llamafactory_guide").r()

        # Create fine-tuning task
        task = create_llm_finetune_tasks(self.dataset, self.model)[1]  # Second task is fine-tuning

        # Set task description using template
        task.description = T("scenarios.finetune.train.prompts:finetuning_task_prompt").r(
            model=self.model,
            dataset=self.dataset,
            runtime_info=runtime_info,
            llamafactory_guide=llamafactory_guide,
        )

        return DSExperiment(pending_tasks_list=[[task]])

    def _execute_experiment(self, exp: DSExperiment, step_name: str):
        """Execute experiment"""
        logger.info(f"Executing {step_name}...")

        if not exp.is_ready_to_run():
            logger.error(f"{step_name} experiment is not ready to run")
            return

        # Final parameter validation for fine-tuning step
        if "Fine-tuning" in step_name:
            self._validate_generated_config(exp)

        # Execute experiment
        workspace = exp.experiment_workspace
        if workspace and hasattr(workspace, "run"):
            result = workspace.run(env=self.env, entry="python main.py")
            logger.info(f"{step_name} execution result: {result.exit_code}")
            if result.stdout:
                logger.info(f"{step_name} output:\n{result.stdout}")
        else:
            logger.warning(f"No executable workspace found for {step_name}")

    def _validate_generated_config(self, exp: DSExperiment):
        """Final validation of generated configuration before execution."""
        try:
            workspace = exp.experiment_workspace
            if workspace and hasattr(workspace, "file_dict"):
                main_py = workspace.file_dict.get("main.py", "")

                # Quick check for known problematic parameters
                problematic_params = [
                    "merge_lora_after_train",
                    "merge_lora",
                    "auto_merge_lora",
                ]
                found_issues = [param for param in problematic_params if param in main_py]

                if found_issues:
                    logger.warning(f"Pre-execution check: Found potentially problematic parameters: {found_issues}")
                    logger.warning(
                        "These parameters may cause training to fail. Consider reviewing the generated configuration.",
                    )
                else:
                    logger.info("Pre-execution parameter validation: No obvious issues detected")

        except Exception as e:
            logger.warning(f"Error during pre-execution validation: {e}")

    def _get_dataset_samples(self) -> str:
        """Get dataset samples (simplified implementation)"""
        try:
            # In Docker environment, dataset is mounted at /workspace/llm_finetune/data/raw
            dataset_path = Path("/workspace/llm_finetune/data/raw") / self.dataset
            if dataset_path.exists():
                # Simple processing, return first few samples
                return f"Dataset path: {dataset_path}\nPlease load and analyze the dataset from this path."
            # Fallback to local path for non-Docker environments
            local_dataset_path = Path(self.ft_rd_setting.local_data_path) / self.dataset
            if local_dataset_path.exists():
                return f"Dataset path: {local_dataset_path}\nPlease load and analyze the dataset from this path."
            return f"Dataset {self.dataset} not found. Please ensure it's available at the expected path."
        except Exception as e:
            logger.warning(f"Could not load dataset samples: {e}")
            return f"Dataset: {self.dataset}\nPlease download and analyze the dataset."
