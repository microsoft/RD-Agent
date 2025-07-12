import unittest


class ConfUtils(unittest.TestCase):

    def test_conf(self):
        import os

        from rdagent.utils.env import EnvConf, QlibDockerConf

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


if __name__ == "__main__":
    unittest.main()
