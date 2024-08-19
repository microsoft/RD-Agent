import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from rdagent.app.kaggle.conf import PROP_SETTING
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import DockerEnv, KGDockerEnv


class KGFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    def execute(self, run_env: dict = {}, *args, **kwargs) -> str:
        qtde = KGDockerEnv(PROP_SETTING.competition)
        qtde.prepare()

        execute_log = qtde.run(
            local_path=str(self.workspace_path),
            entry=f"python train.py",
            env=run_env,
        )

        csv_path = self.workspace_path / "submission.csv"

        if not csv_path.exists():
            logger.error(f"File {csv_path} does not exist.")
            return None
        return pd.read_csv(csv_path, index_col=0).iloc[:, 0]
