import unittest


class ConfUtils(unittest.TestCase):

    def test_conf(self):
        import os

        from rdagent.utils.env import QlibDockerConf

        os.environ["MEM_LIMIT"] = "200g"
        assert QlibDockerConf().mem_limit == "200g"  # base class will affect subclasses
        os.environ["QLIB_DOCKER_MEM_LIMIT"] = "300g"
        assert QlibDockerConf().mem_limit == "300g"  # more accurate subclass will override the base class


if __name__ == "__main__":
    unittest.main()
