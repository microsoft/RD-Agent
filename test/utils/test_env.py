import os
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import shutil

from rdagent.utils.env import LocalConf, LocalEnv, QTDockerEnv

DIRNAME = Path(__file__).absolute().resolve().parent


class EnvUtils(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        # NOTE: For a docker file, the output are generated with root permission.
        # mlrun_p = DIRNAME / "env_tpl" / "mlruns"
        # if mlrun_p.exists():
        #     shutil.rmtree(mlrun_p)
        ...

    # NOTE: Since I don't know the exact environment in which it will be used, here's just an example.
    # NOTE: Because you need to download the data during the prepare process. So you need to have pyqlib in your environment.
    def test_local(self):
        local_conf = LocalConf(
            py_bin="/home/v-linlanglv/miniconda3/envs/RD-Agent-310/bin",
            default_entry="qrun conf.yaml",
        )
        qle = LocalEnv(conf=local_conf)
        qle.prepare()
        conf_path = str(DIRNAME / "env_tpl" / "conf.yaml")
        qle.run(entry="qrun " + conf_path)
        mlrun_p = DIRNAME / "env_tpl" / "mlruns"
        self.assertTrue(mlrun_p.exists(), f"Expected output file {mlrun_p} not found")

    def test_docker(self):
        """
        We will mount `env_tpl` into the docker image.
        And run the docker image with `qrun conf.yaml`
        """
        qtde = QTDockerEnv()
        qtde.prepare()  # you can prepare for multiple times. It is expected to handle it correctly
        # qtde.run("nvidia-smi")  # NOTE: you can check your GPU with this command
        # the stdout are returned as result
        result = qtde.run(local_path=str(DIRNAME / "env_tpl"), entry="qrun conf.yaml")

        mlrun_p = DIRNAME / "env_tpl" / "mlruns"
        self.assertTrue(mlrun_p.exists(), f"Expected output file {mlrun_p} not found")

        # read experiment
        result = qtde.run(local_path=str(DIRNAME / "env_tpl"), entry="python read_exp_res.py")
        print(result)


if __name__ == "__main__":
    unittest.main()
