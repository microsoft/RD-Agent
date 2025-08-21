"""
LLM Fine-tuning Runner Implementation

This module provides a specialized runner for LLM fine-tuning that executes
LLaMA-Factory configuration files instead of Python code.
"""

from pathlib import Path

import yaml

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.data_science.share.eval import ModelDumpEvaluator
from rdagent.components.coder.finetune.conf import FTCoderCoSTEERSettings
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.train.eval import LLMFinetuneEvaluator


class LLMFinetuneRunnerSettings(FTCoderCoSTEERSettings):
    """LLM Fine-tuning specific runner settings."""

    class Config:
        env_prefix = "LLM_FT_Runner_"


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

        # Use a simple evolving strategy that doesn't modify code
        from rdagent.components.coder.CoSTEER.evolving_strategy import (
            MultiProcessEvolvingStrategy,
        )

        es = MultiProcessEvolvingStrategy(scen=scen, settings=settings)

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
        """Execute LLaMA-Factory fine-tuning using the generated configuration."""
        logger.info("Starting LLM fine-tuning execution with LLaMA-Factory")

        try:
            # Get the workspace containing the config.yaml
            workspace = exp.experiment_workspace
            if not workspace or not hasattr(workspace, "file_dict"):
                logger.error("No workspace or file_dict found in experiment")
                return exp

            config_file = workspace.file_dict.get("config.yaml")
            if not config_file:
                logger.error("No config.yaml found in workspace")
                return exp

            # Validate YAML configuration
            try:
                config_dict = yaml.safe_load(config_file)
                logger.info(f"Loaded LLaMA-Factory configuration: {config_dict.get('model_name_or_path', 'Unknown')}")
            except yaml.YAMLError as e:
                logger.error(f"Invalid YAML configuration: {e}")
                return exp

            # Execute LLaMA-Factory training
            result = self._execute_llamafactory_training(workspace, config_dict)

            if result:
                logger.info("LLaMA-Factory training completed successfully")
                # Store training results in experiment
                exp.running_info.running_time = getattr(workspace, "running_info", {}).get("running_time", 0)
            else:
                logger.error("LLaMA-Factory training failed")

            return exp

        except Exception as e:
            logger.error(f"Error during LLM fine-tuning execution: {e}")
            return exp

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
