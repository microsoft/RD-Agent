"""
FT-specific Workspace implementation with minimal checkpoint strategy.

This module provides FTWorkspace, which configures checkpoint to only save
configuration files (train.yaml), excluding all training outputs.

Design Philosophy:
- Checkpoint is for code version control during CoSTEER evolution
- Model persistence is handled separately by Runner's save_model()
- This separation keeps concerns clear and checkpoints lightweight
"""

from typing import TYPE_CHECKING

from rdagent.components.coder.finetune.conf import FT_DATA_FILE_NAME, FT_YAML_FILE_NAME
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import DockerEnv

if TYPE_CHECKING:
    from rdagent.utils.env import Env, EnvResult


class FTWorkspace(FBWorkspace):
    """
    Fine-tuning workspace with minimal checkpoint strategy and unified Docker logging.

    Checkpoint Strategy:
    - Only saves configuration files (train.yaml) for version control
    - Training outputs (models, checkpoints) are excluded by design
    - Final model persistence is Runner's responsibility, not checkpoint's
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Configure checkpoint to save essential files for training
        # Training outputs (models, checkpoints) are managed separately by save_final_model()
        RD_AGENT_SETTINGS.workspace_ckp_white_list_names = [
            FT_YAML_FILE_NAME,      # train.yaml - training config
            FT_DATA_FILE_NAME,      # data.json - processed training data
            "dataset_info.json",    # LlamaFactory dataset config
        ]
        RD_AGENT_SETTINGS.workspace_ckp_size_limit = 100 * 1024

    def run(self, env: "Env", entry: str, env_vars: dict | None = None) -> "EnvResult":
        """Execute the code in the environment with unified Docker logging.

        Args:
            env: The environment to run in (DockerEnv, LocalEnv, etc.)
            entry: The command to execute
            env_vars: Optional additional environment variables (e.g., LLM API keys)
                     Will be merged with default {"PYTHONPATH": "./"}

        Returns:
            EnvResult with stdout, exit_code, running_time
        """
        self.prepare()
        self.inject_files(**self.file_dict)

        # Merge default env with custom env_vars
        run_env = {"PYTHONPATH": "./"}
        if env_vars:
            run_env.update(env_vars)

        result = env.run(entry, str(self.workspace_path), env=run_env)

        # Unified Docker execution logging for FT scenario
        if isinstance(env, DockerEnv):
            logger.log_object(
                {
                    "exit_code": result.exit_code,
                    "stdout": result.stdout or "",
                    "running_time": result.running_time,
                    "entry": entry,
                },
                tag="docker_run.FTWorkspace",
            )

        return result
