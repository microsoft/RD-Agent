import os
import sys
import unittest
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from rdagent.utils.env import QTDockerEnv, LocalEnv, LocalConf
import shutil


DIRNAME = Path(__file__).absolute().resolve().parent


class EnvUtils(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_docker(self):
        """
        We will mount `env_tpl` into the docker image.
        And run the docker image with `qrun conf.yaml`
        """
        qtde = QTDockerEnv()
        qtde.prepare()
        qtde.prepare()  # you can prepare for multiple times. It is expected to handle it correctly
        # the stdout are returned as result
        result = qtde.run(local_path=str(DIRNAME / "env_tpl"), entry="qrun conf2.yaml")
        
        mlrun_p = DIRNAME / "env_tpl" / "mlruns" 
        self.assertTrue(mlrun_p.exists(), f"Expected output file {mlrun_p} not found")

        # read experiment
        result = qtde.run(local_path=str(DIRNAME / "env_tpl"), entry="python read_exp_res.py")
        print("here")
        # print(result)


if __name__ == "__main__":
    unittest.main()
