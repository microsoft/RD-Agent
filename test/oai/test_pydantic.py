import unittest

from rdagent.components.agent.context7 import Agent


class PydanticTest(unittest.TestCase):

    def test_context7(self):
        context7a = Agent()
        res = context7a.query("pandas read_csv encoding error")
        print(res)


if __name__ == "__main__":
    unittest.main()
