import unittest
import nbformat


from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T


class TestTplGen(unittest.TestCase):
    def generate(self, competition: str = "feedback-prize-english-language-learning"):
        
        print(competition)


if __name__ == "__main__":
    unittest.main()
