import unittest

from rdagent.utils.env import DockerEnv, QTDockerEnv

from pathlib import Path
DIRNAME = Path(__file__).absolute().resolve().parent


class TimeUtils(unittest.TestCase):

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
        qtde.run(local_path=DIRNAME / "env_tpl")
        # TODO: test assert: check if the output is generated.
        # - if mlflow is enabled.


if __name__ == "__main__":
    unittest.main()
