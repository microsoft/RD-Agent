"""
LLM Fine-tuning Runner Implementation

This module provides a specialized runner for LLM fine-tuning that executes
LLaMA-Factory configuration files instead of Python code.
"""

from pathlib import Path

import yaml

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiEvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    EvolvingStrategy,
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.components.coder.data_science.share.eval import ModelDumpEvaluator
from rdagent.components.coder.finetune.conf import FTCoderCoSTEERSettings
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.train.eval import LLMFinetuneEvaluator


class LLMFinetuneRunnerSettings(FTCoderCoSTEERSettings):
    """LLM Fine-tuning specific runner settings."""

    class Config:
        env_prefix = "LLM_FT_Runner_"


class LLMFinetuneRunnerEvolvingStrategy(MultiProcessEvolvingStrategy):
    """Evolving strategy for LLM fine-tuning runner.

    This strategy modifies the config to run on full dataset instead of debug mode.
    """

    def implement_one_task(
        self,
        target_task: Task,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        """Modify config for full dataset training."""

        if not workspace or "config.yaml" not in workspace.file_dict:
            logger.error("No config.yaml found in workspace")
            return {}

        # Load existing config from workspace
        config_content = workspace.file_dict["config.yaml"]
        try:
            config_dict = yaml.safe_load(config_content)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config.yaml: {e}")
            return {}

        # Modify config for full dataset training
        # Remove debug mode settings
        if "max_samples" in config_dict:
            del config_dict["max_samples"]

        # Increase training epochs for full training
        if "num_train_epochs" in config_dict:
            config_dict["num_train_epochs"] = max(3, config_dict.get("num_train_epochs", 3))

        # Adjust batch size for full training if needed
        if "per_device_train_batch_size" in config_dict:
            # Keep the same batch size or adjust based on GPU memory
            pass

        # Update output directory to indicate full training
        if "output_dir" in config_dict:
            config_dict["output_dir"] = config_dict["output_dir"].replace("debug", "full")

        # Convert back to YAML
        new_config_yaml = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

        return {
            "config.yaml": new_config_yaml,
            self.KEY_CHANGE_SUMMARY: "Modified configuration for full dataset training: removed sample limit, adjusted epochs",
        }

    def assign_code_list_to_evo(self, code_list: list[dict], evo: EvolvingItem) -> None:
        """Assign the modified config to the evolving item."""
        if not code_list:
            return

        # For runner, we only have one task and one code modification
        code_dict = code_list[0] if code_list else {}

        # Update the workspace with new config
        if evo.workspace and code_dict:
            for filename, content in code_dict.items():
                if filename != self.KEY_CHANGE_SUMMARY:
                    evo.workspace.file_dict[filename] = content


class LLMFinetuneRunner(CoSTEER):
    """LLM Fine-tuning specific runner that executes LLaMA-Factory configurations."""

    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        # Use LLM fine-tuning specific evaluator
        eval_l = [LLMFinetuneEvaluator(scen=scen)]

        # Add model dump evaluator if enabled
        if FT_RD_SETTING.enable_model_dump:
            eval_l.append(ModelDumpEvaluator(scen=scen, data_type="full"))

        eva = CoSTEERMultiEvaluator(single_evaluator=eval_l, scen=scen)
        settings = LLMFinetuneRunnerSettings()

        # Use runner-specific evolving strategy for full dataset training
        es = LLMFinetuneRunnerEvolvingStrategy(scen=scen, settings=settings)

        # Initialize with LLM-specific configuration
        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            max_loop=getattr(FT_RD_SETTING, "runner_max_loop", 1),  # Default to 1 loop for running
            **kwargs,
        )

    def develop(self, exp):
        """Execute LLaMA-Factory fine-tuning on full dataset.

        This method runs the full training workflow:
        1. Modify config for full dataset training
        2. Execute training
        3. Evaluate the trained model
        """
        logger.info("Starting full dataset LLM fine-tuning with LLaMA-Factory")

        # Run the standard CoSTEER develop process which will:
        # 1. Use LLMFinetuneRunnerEvolvingStrategy to modify config for full training
        # 2. Execute the training
        # 3. Evaluate using LLMFinetuneEvaluator
        exp = super().develop(exp)

        # Additional post-training evaluation can be added here if needed
        if hasattr(exp, "experiment_workspace") and exp.experiment_workspace:
            self._evaluate_trained_model(exp.experiment_workspace)

        return exp

    def _evaluate_trained_model(self, workspace: FBWorkspace):
        """Evaluate the trained model performance.

        TODO: Implement specific evaluation metrics for AIME2024/AIME2025 when available.
        Currently uses basic evaluation from LLMFinetuneEvaluator.
        """
        logger.info("Evaluating trained model performance...")

        # Check for model output files
        model_files = list(workspace.workspace_path.glob("output/adapter_model.bin"))
        if model_files:
            logger.info(f"Found trained model files: {[f.name for f in model_files]}")

            # TODO: Add specific evaluation logic here
            # For now, the evaluation is handled by LLMFinetuneEvaluator in the CoSTEER framework
        else:
            logger.warning("No trained model files found for evaluation")

    def _execute_llamafactory_training(self, workspace: FBWorkspace, config_dict: dict) -> bool:
        """Execute LLaMA-Factory training using the configuration."""
        try:
            # Save config.yaml to workspace
            config_path = workspace.workspace_path / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)

            # Create LLaMA-Factory command
            cmd = [
                "llamafactory-cli",
                "train",
                "--config",
                str(config_path),
                "--output_dir",
                str(workspace.workspace_path / "output"),
            ]

            # Execute training command
            import subprocess

            logger.info(f"Executing LLaMA-Factory command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, cwd=workspace.workspace_path, capture_output=True, text=True, timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info("LLaMA-Factory training completed successfully")
                logger.info(f"Training output: {result.stdout}")
                return True
            else:
                logger.error(f"LLaMA-Factory training failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("LLaMA-Factory training timed out")
            return False
        except Exception as e:
            logger.error(f"Error executing LLaMA-Factory training: {e}")
            return False

    def get_develop_max_seconds(self) -> int | None:
        """Get maximum seconds for development using FT settings."""
        return int(self.scen.real_full_timeout() * self.settings.max_seconds_multiplier)

    def compare_and_pick_fb(self, base_fb, new_fb) -> bool:
        """Compare feedback for LLM fine-tuning results."""
        if base_fb is None:
            return True

        base_fb = base_fb[0]
        new_fb = new_fb[0]

        def compare_scores(s1, s2) -> bool:
            if s2 is None:
                return False
            if s1 is None:
                return True
            return (s2 > s1) == self.scen.metric_direction

        return compare_scores(getattr(base_fb, "score", None), getattr(new_fb, "score", None))
