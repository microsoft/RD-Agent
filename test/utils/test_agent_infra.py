import unittest

from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T


class TestAgentInfra(unittest.TestCase):
    def test_agent_infra(self):
        # NOTE: It is not serious. It is just for testing
        sys_prompt = T("components.proposal.prompts:hypothesis_gen.system_prompt").r(
            targets="targets",
            scenario=T("scenarios.qlib.experiment.prompts:qlib_model_background").r(),
            hypothesis_output_format=PythonAgentOut.get_spec(),
            hypothesis_specification=PythonAgentOut.get_spec(),
        )
        user_prompt = T("components.proposal.prompts:hypothesis_gen.user_prompt").r(
            hypothesis_and_feedback="No Feedback",
            RAG="No RAG",
            targets="targets",
        )
        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt=user_prompt, system_prompt=sys_prompt)
        code = PythonAgentOut.extract_output(resp)

        print(code)


if __name__ == "__main__":
    unittest.main()
