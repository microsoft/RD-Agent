from pathlib import Path
import pandas as pd

from rdagent.app.data_mining.conf import PROP_SETTING
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import KGDockerEnv, DockerEnv, KaggleConf

import subprocess
import zipfile

class KGFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    def execute(self, run_env: dict = {}, *args, **kwargs) -> str:
        qtde = KGDockerEnv()
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

class KaggleFBWorkspace(FBWorkspace):
    def __init__(self,
                 competition: str,
                 shared_data_path: str = "/data/userdata/share/kaggle",
                 *args,
                 **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.competition = competition

        # download data
        local_data_path = f"{shared_data_path}/{competition}"
        subprocess.run(["kaggle", "competitions", "download", "-c", competition, "-p", local_data_path])

        # unzip data
        with zipfile.ZipFile(f"{local_data_path}/{competition}.zip", 'r') as zip_ref:
            zip_ref.extractall(local_data_path)
        
        self.kaggle_conf = KaggleConf()
        # mount kaggle input data
        self.kaggle_conf.extra_volumes = {local_data_path: f"/kaggle/input/{competition}"}

    def execute(self, *args, **kwargs) -> str:
        # TODO: run codes of the kaggle competition
        kaggle_env = DockerEnv(self.kaggle_conf)
        kaggle_env.prepare()

        kaggle_env.run(entry=f"ls /kaggle/input/{self.competition}")