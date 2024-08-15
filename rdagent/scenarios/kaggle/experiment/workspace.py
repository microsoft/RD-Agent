from rdagent.utils.env import DockerEnv, KaggleConf
from rdagent.core.experiment import FBWorkspace

import subprocess
import zipfile

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