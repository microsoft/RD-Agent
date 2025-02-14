import json
import unittest

from rdagent.oai.llm_utils import APIBackend


class TestChatCompletion(unittest.TestCase):
    def test_chat_completion(self) -> None:
        system_prompt = "You are a helpful assistant."
        user_prompt = "What is your name?"
        response = APIBackend().build_messages_and_create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        assert response is not None
        assert isinstance(response, str)

    def test_chat_completion_json_mode(self) -> None:
        system_prompt = "You are a helpful assistant. answer in Json format."
        user_prompt = "What is your name?"
        response = APIBackend().build_messages_and_create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
        )
        assert response is not None
        assert isinstance(response, str)
        json.loads(response)

    def test_build_messages_and_calculate_token(self) -> None:
        system_prompt = "You are a helpful assistant."
        user_prompt = "What is your name?"
        token = APIBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
        assert token is not None
        assert isinstance(token, int)


if __name__ == "__main__":
    unittest.main()
