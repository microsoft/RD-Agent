import os
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import shutil

from rdagent.utils.env import LocalConf, LocalEnv, QlibDockerConf, QTDockerEnv

DIRNAME = Path(__file__).absolute().resolve().parent


class EnvUtils(unittest.TestCase):
    def setUp(self):
        self.test_workspace = DIRNAME / "test_workspace"
        self.test_workspace.mkdir(exist_ok=True)

    def tearDown(self):
        if self.test_workspace.exists():
            shutil.rmtree(self.test_workspace)

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
        """We will mount `env_tpl` into the docker image.
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

    def test_run_ret_code(self):
        """Test the run_ret_code method of QTDockerEnv with both valid and invalid commands."""
        qtde = QTDockerEnv()
        qtde.prepare()

        # Test with a valid command
        result, return_code = qtde.run_ret_code(entry='echo "Hello, World!"', local_path=str(self.test_workspace))
        print(return_code)
        assert return_code == 0, f"Expected return code 0, but got {return_code}"
        assert "Hello, World!" in result, "Expected output not found in result"

        # Test with an invalid command
        _, return_code = qtde.run_ret_code(entry="invalid_command", local_path=str(self.test_workspace))
        print(return_code)
        assert return_code != 0, "Expected non-zero return code for invalid command"

        dc = QlibDockerConf()
        dc.running_timeout_period = 1
        qtde = QTDockerEnv(dc)
        result, return_code = qtde.run_ret_code(entry="sleep 2", local_path=str(self.test_workspace))
        print(result)
        assert return_code == 124, "Expected return code 124 for timeout"

    def test_docker_mem(self):
        cmd = 'python -c \'print("start"); import numpy as np;  size_mb = 500; size = size_mb * 1024 * 1024 // 8; array = np.random.randn(size).astype(np.float64); print("success")\''

        qtde = QTDockerEnv(QlibDockerConf(mem_limit="10m"))
        qtde.prepare()
        result = qtde.run(local_path=str(DIRNAME / "env_tpl"), entry=cmd)
        self.assertTrue(not result.strip().endswith("success"))

        qtde = QTDockerEnv(QlibDockerConf(mem_limit="1g"))
        qtde.prepare()
        result = qtde.run(local_path=str(DIRNAME / "env_tpl"), entry=cmd)
        self.assertTrue(result.strip().endswith("success"))

        # The above command equals to the follow commands with dockr cli.sh
        # docker run  --memory=10m  -it --rm local_qlib:latest python -c 'import numpy as np; print(123);  size_mb = 1; size = size_mb * 1024 * 1024 // 8; array = np.random.randn(size).astype(np.float64); array[0], array[-1] = 1.0, 1.0; print(321)'
        # docker run  --memory=10g  -it --rm local_qlib:latest python -c 'import numpy as np; print(123);  size_mb = 1; size = size_mb * 1024 * 1024 // 8; array = np.random.randn(size).astype(np.float64); array[0], array[-1] = 1.0, 1.0; print(321)'


if __name__ == "__main__":
    unittest.main()
