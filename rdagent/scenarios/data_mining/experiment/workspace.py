from pathlib import Path

import pandas as pd

from rdagent.app.data_mining.conf import PROP_SETTING
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import DMDockerEnv


class DMFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    def execute(self, run_env: dict = {}, *args, **kwargs) -> str:
        qtde = DMDockerEnv()
        qtde.prepare(PROP_SETTING.username, PROP_SETTING.password)

        execute_log = qtde.run(
            local_path=str(self.workspace_path),
            entry=f"python train.py",
            env=run_env,
        )

        csv_path = self.workspace_path / "submission.txt"

        if not csv_path.exists():
            logger.error(f"File {csv_path} does not exist.")
            return None
        with open(self.workspace_path / "submission.txt", "r") as f:
            return f.read()
