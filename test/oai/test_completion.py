import json
import unittest
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

from rdagent.oai.llm_utils import APIBackend


class TestPersonModel(BaseModel):
    """This is a test Pydantic model"""

    name: str = Field(description="name")
    age: int = Field(description="age")
    skills: List[str] = Field(description="skills")


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

    def test_json_mode_with_specific_target_type(self) -> None:
        """Test json_mode=True with specific json_target_type"""
        system_prompt = "You are a helpful assistant. Please respond according to requirements."
        user_prompt = "Generate programmer information including name, age, and skills list"

        response = APIBackend().build_messages_and_create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            json_target_type=Dict[str, Union[str, int, List[str]]],
        )

        # Verify response format
        assert response is not None
        assert isinstance(response, str)

        # Verify JSON format
        parsed = json.loads(response)
        assert isinstance(parsed, dict)

    def test_response_format_with_basemodel(self) -> None:
        """Test response_format with BaseModel (if supported)"""
        backend = APIBackend()

        system_prompt = "You are a helpful assistant. Please respond according to requirements."
        user_prompt = "Generate programmer information including name, age, and skills list"

        if backend.supports_response_schema():
            # Use BaseModel when response_schema is supported
            response = backend.build_messages_and_create_chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=TestPersonModel,
            )
        else:
            # Use dict + json_target_type when not supported
            response = backend.build_messages_and_create_chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format={"type": "json_object"},
                json_target_type=Dict[str, Union[str, int, List[str]]],
            )

        # Verify response format
        assert response is not None
        assert isinstance(response, str)

        # Verify JSON format
        parsed = json.loads(response)
        assert isinstance(parsed, dict)


if __name__ == "__main__":
    unittest.main()
