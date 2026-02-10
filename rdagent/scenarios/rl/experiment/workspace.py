"""
RL Post-training Workspace

参考 SFT: rdagent/scenarios/finetune/experiment/workspace.py
"""

from pathlib import Path
from typing import TYPE_CHECKING

from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger

if TYPE_CHECKING:
    from rdagent.utils.env import Env

from rdagent.utils.env import DockerEnv, EnvResult


class RLWorkspace(FBWorkspace):
    """RL 训练工作区"""

    def run(self, env: "Env", entry: str) -> EnvResult:
        """在环境中执行命令"""
        self.prepare()
        self.inject_files(**self.file_dict)
        
        result = env.run(entry, str(self.workspace_path))
        
        tag_prefix = "docker_run" if isinstance(env, DockerEnv) else "env_run"
        logger.log_object(
            {
                "exit_code": result.exit_code,
                "stdout": result.stdout or "",
                "running_time": result.running_time,
                "entry": entry,
                "workspace_path": str(self.workspace_path),
            },
            tag=f"{tag_prefix}.RLWorkspace",
        )
        
        return result

