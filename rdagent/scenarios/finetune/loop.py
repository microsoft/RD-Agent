"""
LLM Fine-tuning Loop Implementation

Optimized LLM fine-tuning loop that preprocesses data during initialization
and focuses on model fine-tuning in the main loop.
"""

from pathlib import Path

from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.finetune.data_process.data_format_converter import (
    DataFormatConverter,
)
from rdagent.scenarios.finetune.tasks import create_llm_finetune_tasks

# Import LLM-specific components
from rdagent.scenarios.finetune.train.coder import LLMPipelineCoSTEER
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T


class LLMFinetuneRDLoop:
    """LLM fine-tuning loop with data preprocessing during initialization"""

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

        # Import get_ft_env from finetune coder configuration
        from rdagent.components.coder.finetune.conf import get_ft_env

        self.env = get_ft_env(
            extra_volumes=data_volumes,
            running_timeout_period=None,  # No time limit
            enable_cache=False,
        )

        logger.info(f"Initialized LLM finetune loop for {model} on {dataset}")
        logger.info(f"Shared workspace: {self.shared_workspace_dir}")

        # Preprocess data during initialization
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess and convert dataset format during initialization"""
        logger.info("Preprocessing dataset format...")

        # Create data format converter
        data_converter = DataFormatConverter(
            dataset=self.dataset, model=self.model, ft_rd_setting=self.ft_rd_setting, scen=self.scen
        )

        # Convert dataset format
        success = data_converter.convert_dataset(self.env, self.shared_workspace_dir)

        if not success:
            raise RuntimeError("Failed to preprocess dataset. Cannot proceed with fine-tuning.")

        logger.info("Dataset preprocessing completed successfully")

    def run(self):
        """Run LLM fine-tuning"""
        logger.info("Starting LLM fine-tuning...")

        # Data has already been preprocessed in __init__
        # Now focus on model fine-tuning
        finetune_exp = self._create_finetuning_experiment()
        finetune_exp = self.coder.develop(finetune_exp)
        self._execute_experiment(finetune_exp)

        logger.info("LLM fine-tuning completed!")

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

    def _execute_experiment(self, exp: DSExperiment):
        """Execute fine-tuning experiment"""
        logger.info("Executing fine-tuning experiment...")

        if not exp.is_ready_to_run():
            logger.error("Fine-tuning experiment is not ready to run")
            return

        # Final parameter validation before execution
        self._validate_generated_config(exp)

        # Execute experiment
        workspace = exp.experiment_workspace
        if workspace and hasattr(workspace, "run"):
            result = workspace.run(env=self.env, entry="python main.py")
            logger.info(f"Fine-tuning execution result: {result.exit_code}")
            if result.stdout:
                logger.info(f"Fine-tuning output:\n{result.stdout}")
        else:
            logger.warning("No executable workspace found for fine-tuning")

    def _validate_generated_config(self, exp: DSExperiment):
        # TODO: move to coder evaluation
        """Final validation of generated configuration using LLamaFactory official parameter validator."""
        try:
            import re

            import yaml

            from rdagent.scenarios.finetune.train.utils import (
                create_parameter_validator,
            )

            validator = create_parameter_validator()
            workspace = exp.experiment_workspace

            if workspace and hasattr(workspace, "file_dict"):
                main_py = workspace.file_dict.get("main.py", "")
                validation_warnings = []

                # Extract and validate YAML configurations from the code
                yaml_patterns = [
                    r"yaml\.dump\s*\(\s*(\{[^}]+\})",  # yaml.dump({...})
                    r"config\s*=\s*(\{[^}]+\})",  # config = {...}
                    r"training_args\s*=\s*(\{[^}]+\})",  # training_args = {...}
                ]

                for pattern in yaml_patterns:
                    matches = re.finditer(pattern, main_py, re.DOTALL)
                    for match in matches:
                        try:
                            # Extract and safely evaluate configuration dictionary
                            config_str = match.group(1)
                            # Convert Python dict format to evaluable format
                            config_str = re.sub(r"\s+", " ", config_str.strip())

                            config_dict = eval(config_str)  # Safe since we control the pattern
                            if isinstance(config_dict, dict):
                                # Validate using official LLamaFactory validator
                                original_param_count = len(config_dict)
                                validated_config = validator.validate_config_dict(config_dict)
                                validated_param_count = len(validated_config)

                                if validated_param_count < original_param_count:
                                    invalid_params = set(config_dict.keys()) - set(validated_config.keys())
                                    warning_msg = f"Found {len(invalid_params)} invalid LLamaFactory parameters: {list(invalid_params)}"
                                    validation_warnings.append(warning_msg)

                        except Exception as e:
                            logger.debug(f"Could not parse config pattern: {e}")
                            continue

                # Report validation results
                if validation_warnings:
                    logger.warning("Pre-execution validation completed with warnings:")
                    for warning in validation_warnings:
                        logger.warning(f"  - {warning}")
                    logger.warning("Training may encounter issues with these parameters.")
                else:
                    logger.info("Pre-execution parameter validation: All parameters appear valid")

        except Exception as e:
            logger.warning(f"Error during pre-execution validation: {e}")
