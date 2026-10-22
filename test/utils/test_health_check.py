import os
from unittest.mock import patch

import pytest

from rdagent.app.utils import health_check


@pytest.mark.offline
@pytest.mark.parametrize(
    ("embedding_env", "expected_key", "expected_base"),
    [
        ({}, "chat-key", "https://chat.example/v1"),
        (
            {"LITELLM_PROXY_API_KEY": "embedding-key", "LITELLM_PROXY_API_BASE": "https://embedding.example/v1"},
            "embedding-key",
            "https://embedding.example/v1",
        ),
        ({"LITELLM_PROXY_API_KEY": "embedding-key"}, "embedding-key", "https://chat.example/v1"),
        ({"LITELLM_PROXY_API_BASE": "https://embedding.example/v1"}, "chat-key", "https://embedding.example/v1"),
        (
            {"LITELLM_PROXY_API_KEY": "", "LITELLM_PROXY_API_BASE": ""},
            "chat-key",
            "https://chat.example/v1",
        ),
    ],
)
def test_openai_health_check_embedding_credentials(
    embedding_env: dict[str, str], expected_key: str, expected_base: str
) -> None:
    env = {
        "OPENAI_API_KEY": "chat-key",
        "OPENAI_API_BASE": "https://chat.example/v1",
        "CHAT_MODEL": "openai/chat-model",
        "EMBEDDING_MODEL": "litellm_proxy/embedding-model" if embedding_env else "text-embedding-3-small",
        **embedding_env,
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(health_check, "embedding") as embedding,
        patch.object(health_check, "completion") as completion,
    ):
        health_check.env_check()

    embedding.assert_called_once_with(
        model=env["EMBEDDING_MODEL"], api_key=expected_key, api_base=expected_base, input="Hello world!"
    )
    completion.assert_called_once_with(
        model=env["CHAT_MODEL"],
        api_key=env["OPENAI_API_KEY"],
        api_base=env["OPENAI_API_BASE"],
        messages=[{"role": "user", "content": "Hello!"}],
    )


@pytest.mark.offline
@pytest.mark.parametrize("chat_base", [None, "https://deepseek.example/v1"])
def test_deepseek_health_check_keeps_separate_embedding_credentials(chat_base: str | None) -> None:
    env = {
        "DEEPSEEK_API_KEY": "deepseek-key",
        "CHAT_MODEL": "deepseek/deepseek-chat",
        "EMBEDDING_MODEL": "litellm_proxy/embedding-model",
        "LITELLM_PROXY_API_KEY": "embedding-key",
        "LITELLM_PROXY_API_BASE": "https://embedding.example/v1",
    }
    if chat_base is not None:
        env["DEEPSEEK_API_BASE"] = chat_base
    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(health_check, "embedding") as embedding,
        patch.object(health_check, "completion") as completion,
    ):
        health_check.env_check()

    embedding.assert_called_once_with(
        model=env["EMBEDDING_MODEL"],
        api_key=env["LITELLM_PROXY_API_KEY"],
        api_base=env["LITELLM_PROXY_API_BASE"],
        input="Hello world!",
    )
    expected_chat_kwargs = {
        "model": env["CHAT_MODEL"],
        "api_key": env["DEEPSEEK_API_KEY"],
        "messages": [{"role": "user", "content": "Hello!"}],
    }
    if chat_base is not None:
        expected_chat_kwargs["api_base"] = chat_base
    completion.assert_called_once_with(**expected_chat_kwargs)
