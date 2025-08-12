import os
import unittest

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.conf import DSCoderCoSTEERSettings
from rdagent.scenarios.data_science.dev.runner import DSRunnerCoSTEERSettings
from rdagent.utils.env import EnvConf, QlibDockerConf


class ConfUtils(unittest.TestCase):

    def test_conf(self):

        os.environ["MEM_LIMIT"] = "200g"
        os.environ["RUNNING_TIMEOUT_PERIOD"] = "None"
        assert QlibDockerConf().mem_limit == "200g"  # base class will affect subclasses
        os.environ["QLIB_DOCKER_MEM_LIMIT"] = "300g"
        assert QlibDockerConf().mem_limit == "300g"  # more accurate subclass will override the base class
        assert QlibDockerConf().running_timeout_period is None

        os.environ["DEFAULT_ENTRY"] = "which python"
        os.environ["ENABLE_CACHE"] = "False"

        assert EnvConf().enable_cache is False
        assert QlibDockerConf().enable_cache is False

        os.environ["ENABLE_CACHE"] = "True"
        assert EnvConf().enable_cache is True
        assert QlibDockerConf().enable_cache is True

    def test_ds_costeer_conf(self):
        os.environ["DS_CODER_COSTEER_MAX_SECONDS_MULTIPLIER"] = "1000"
        coder_conf = DSCoderCoSTEERSettings()
        runner_conf = DSRunnerCoSTEERSettings()
        print(coder_conf.max_seconds_multiplier)
        print(runner_conf.max_seconds_multiplier)
        assert coder_conf.max_seconds_multiplier == 1000
        # NOTE: coder's config should not affect runner's config
        assert runner_conf.max_seconds_multiplier == 1
        os.environ["DS_RUNNER_COSTEER_MAX_SECONDS"] = "2000"
        assert DSRunnerCoSTEERSettings().max_seconds == 2000


if __name__ == "__main__":
    unittest.main()
