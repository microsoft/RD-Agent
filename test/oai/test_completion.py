import json
import random
import unittest

from rdagent.oai.llm_utils import APIBackend


class TestChatCompletion(unittest.TestCase):
    def test_chat_completion(self) -> None:
        system_prompt = "You are a helpful assistant."
        user_prompt = "What is your name?"
        response = APIBackend().build_messages_and_create_chat_completion(
            system_prompt=system_prompt, user_prompt=user_prompt,
        )
        assert response is not None
        assert isinstance(response, str)

    def test_chat_completion_json_mode(self) -> None:
        system_prompt = "You are a helpful assistant. answer in Json format."
        user_prompt = "What is your name?"
        response = APIBackend().build_messages_and_create_chat_completion(
            system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True,
        )
        assert response is not None
        assert isinstance(response, str)
        json.loads(response)

    def test_chat_multi_round(self) -> None:
        system_prompt = "You are a helpful assistant."
        fruit_name = random.SystemRandom().choice(["apple", "banana", "orange", "grape", "watermelon"])
        user_prompt_1 = (
            f"I will tell you a name of fruit, please remember them and tell me later. "
            f"The name is {fruit_name}. Once you remember it, please answer OK."
        )
        user_prompt_2 = "What is the name of the fruit I told you before?"

        session = APIBackend().build_chat_session(session_system_prompt=system_prompt)

        response_1 = session.build_chat_completion(user_prompt=user_prompt_1)
        assert response_1 is not None
        assert "ok" in response_1.lower()
        response2 = session.build_chat_completion(user_prompt=user_prompt_2)
        assert response2 is not None


if __name__ == "__main__":
    unittest.main()
