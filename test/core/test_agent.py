import unittest

from rdagent.core.agent import Agent


class GrammarAgent(Agent):
    ...


class TestAgent(unittest.TestCase):

    def test_ga(self):
        ga = GrammarAgent()
        # TODO: we have no api,  so we skip the test
        # print(ga.call("I is a student", context={"context": ""}))


if __name__ == "__main__":
    unittest.main()
