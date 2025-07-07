import os
import sys
import time
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import shutil

from rdagent.utils.env import (
    CondaConf,
    LocalConf,
    LocalEnv,
    QlibDockerConf,
    QTDockerEnv,
    cleanup_container,
)

DIRNAME = Path(__file__).absolute().resolve().parent


class QlibLocalEnv(LocalEnv):
    def prepare(self) -> None:
        if not (Path("~/.qlib/qlib_data/cn_data").expanduser().resolve().exists()):
            self.check_output(
                entry="python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn",
            )
        else:
            print("Data already exists. Download skipped.")


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
            bin_path="/home/v-linlanglv/miniconda3/envs/RD-Agent-310/bin",
            default_entry="qrun conf.yaml",
        )
        qle = QlibLocalEnv(conf=local_conf)
        qle.prepare()
        conf_path = str(DIRNAME / "env_tpl" / "conf.yaml")
        qle.check_output(entry="qrun " + conf_path)
        mlrun_p = DIRNAME / "env_tpl" / "mlruns"
        self.assertTrue(mlrun_p.exists(), f"Expected output file {mlrun_p} not found")

    def test_local_simple(self):
        code_path = DIRNAME / "tmp_code"
        code_path.mkdir(exist_ok=True)
        # Get user home dynamically
        home_bin = str(Path.home() / "miniconda3/bin/")
        local_conf = LocalConf(bin_path=home_bin, default_entry="which python")

        local_conf.extra_volumes = {str(code_path): "./code"}
        print(local_conf)
        le = LocalEnv(conf=local_conf)
        le.prepare()
        result = le.run(local_path=str(code_path))
        print(result.stdout, result.exit_code, result.running_time)

    def test_conda_simple(self):
        conda_conf = CondaConf(default_entry="which python", conda_env_name="MLE")
        le = LocalEnv(conf=conda_conf)
        le.prepare()
        code_path = DIRNAME / "tmp_code"
        code_path.mkdir(exist_ok=True)
        result = le.run(local_path=str(code_path))
        print(result.stdout, result.exit_code, result.running_time)

    def test_conda_error(self):
        conda_conf = CondaConf(conda_env_name="MLE")
        le = LocalEnv(conf=conda_conf)
        le.prepare()
        file_name = f"{time.time()}.py"
        with open(self.test_workspace / file_name, "w") as f:
            f.write('import json \njson.loads(b\'{"name": "\xa1"}\')')
        result = le.run(local_path=str(self.test_workspace), entry=f"python {file_name}")
        assert result.exit_code == 1
        assert "bytes can only contain ASCII literal characters" in result.stdout

    def test_docker(self):
        """We will mount `env_tpl` into the docker image.
        And run the docker image with `qrun conf.yaml`
        """
        qtde = QTDockerEnv()
        qtde.prepare()  # you can prepare for multiple times. It is expected to handle it correctly
        # qtde.run("nvidia-smi")  # NOTE: you can check your GPU with this command
        # the stdout are returned as result
        result = qtde.check_output(local_path=str(DIRNAME / "env_tpl"), entry="qrun conf.yaml")

        mlrun_p = DIRNAME / "env_tpl" / "mlruns"
        self.assertTrue(mlrun_p.exists(), f"Expected output file {mlrun_p} not found")

        # read experiment
        result = qtde.check_output(local_path=str(DIRNAME / "env_tpl"), entry="python read_exp_res.py")
        print(result)

    def test_run(self):
        """Test the run method of QTDockerEnv with both valid and invalid commands."""
        qtde = QTDockerEnv()
        qtde.prepare()

        # Test with a valid command
        result = qtde.run(entry='echo "Hello, World!"', local_path=str(self.test_workspace))
        print(result.exit_code)
        assert result.exit_code == 0, f"Expected return code 0, but got {result.exit_code}"
        assert "Hello, World!" in result.stdout, "Expected output not found in result"

        # Test with an invalid command
        result = qtde.run(entry="invalid_command", local_path=str(self.test_workspace))
        print(result.exit_code)
        assert result.exit_code != 0, "Expected non-zero return code for invalid command"

        dc = QlibDockerConf()
        dc.running_timeout_period = 1
        qtde = QTDockerEnv(dc)
        result = qtde.run(entry="sleep 2", local_path=str(self.test_workspace))
        print(result.exit_code)
        assert result.exit_code == 124, "Expected return code 124 for timeout"

    def test_docker_mem(self):
        cmd = 'python -c \'print("start"); import numpy as np;  size_mb = 500; size = size_mb * 1024 * 1024 // 8; array = np.random.randn(size).astype(np.float64); print("success")\''

        qtde = QTDockerEnv(QlibDockerConf(mem_limit="10m"))
        qtde.prepare()
        result = qtde.check_output(local_path=str(DIRNAME / "env_tpl"), entry=cmd)
        self.assertTrue(not result.strip().endswith("success"))

        qtde = QTDockerEnv(QlibDockerConf(mem_limit="1g"))
        qtde.prepare()
        result = qtde.check_output(local_path=str(DIRNAME / "env_tpl"), entry=cmd)
        self.assertTrue(result.strip().endswith("success"))

        # The above command equals to the follow commands with dockr cli.sh
        # docker run  --memory=10m  -it --rm local_qlib:latest python -c 'import numpy as np; print(123);  size_mb = 1; size = size_mb * 1024 * 1024 // 8; array = np.random.randn(size).astype(np.float64); array[0], array[-1] = 1.0, 1.0; print(321)'
        # docker run  --memory=10g  -it --rm local_qlib:latest python -c 'import numpy as np; print(123);  size_mb = 1; size = size_mb * 1024 * 1024 // 8; array = np.random.randn(size).astype(np.float64); array[0], array[-1] = 1.0, 1.0; print(321)'

    def test_cleanup_container_import(self):
        """Test that cleanup_container function can be imported and has correct interface."""
        # Test that the function exists and can be called
        self.assertTrue(callable(cleanup_container))

        # Test with None (should not raise an exception)
        cleanup_container(None, "test context")

        # The function should accept positional and keyword arguments
        cleanup_container(None)
        cleanup_container(None, context="test")


if __name__ == "__main__":
    unittest.main()
