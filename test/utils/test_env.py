import os
import sys
import subprocess
import unittest
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from rdagent.utils.env import QTDockerEnv


DIRNAME = Path(__file__).absolute().resolve().parent


class EnvUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_docker(self):
        """
        We will mount `env_tpl` into the docker image.
        And run the docker image with `qrun conf.yaml`
        """
        qtde = QTDockerEnv()
        qtde.prepare()
        # qtde.run(local_path=DIRNAME / "env_tpl")
        # TODO: test assert: check if the output is generated.
        # - if mlflow is enabled.
        output_dir = DIRNAME / "env_tpl" / "output"
        if not output_dir.exists():
            os.makedirs(output_dir)
        yaml_path = DIRNAME / "env_tpl" / "conf.yaml"
        result = subprocess.run(['pwd'], capture_output=True, text=True, check=True)
        pwd_output = result.stdout.strip()
        print(pwd_output)
        qtde.run(local_path=str(DIRNAME / "env_tpl"), entry="qrun test/utils/env_tpl/conf.yaml", extra_volumes=extra_volumes)

        output_file = output_dir / "output_file"
        self.assertTrue(output_file.exists(), f"Expected output file {output_file} not found")


if __name__ == "__main__":
    unittest.main()
