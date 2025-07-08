from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from rdagent.core.conf import ExtendedBaseSettings


class LLMSettings(ExtendedBaseSettings):
    # backend
    backend: str = "rdagent.oai.backend.LiteLLMAPIBackend"

    chat_model: str = "gpt-4-turbo"
    embedding_model: str = "text-embedding-3-small"

    reasoning_effort: Literal["low", "medium", "high"] | None = None
    enable_response_schema: bool = True
    # Whether to enable response_schema in chat models. may not work for models that do not support it.

    # Handling format
    reasoning_think_rm: bool = False
    """
    Some LLMs include <think>...</think> tags in their responses, which can interfere with the main output.
    Set reasoning_think_rm to True to remove any <think>...</think> content from responses.
    """

    # TODO: most of the settings are only used on deprec.DeprecBackend.
    # So they should move the settings to that folder.

    log_llm_chat_content: bool = True

    use_azure: bool = Field(default=False, deprecated=True)
    chat_use_azure: bool = False
    embedding_use_azure: bool = False

    chat_use_azure_token_provider: bool = False
    embedding_use_azure_token_provider: bool = False
    managed_identity_client_id: str | None = None
    max_retry: int = 10
    retry_wait_seconds: int = 1
    dump_chat_cache: bool = False
    use_chat_cache: bool = False
    dump_embedding_cache: bool = False
    use_embedding_cache: bool = False
    prompt_cache_path: str = str(Path.cwd() / "prompt_cache.db")
    max_past_message_include: int = 10
    timeout_fail_limit: int = 10
    violation_fail_limit: int = 1

    # Behavior of returning answers to the same question when caching is enabled
    use_auto_chat_cache_seed_gen: bool = False
    """
    `_create_chat_completion_inner_function` provides a feature to pass in a seed to affect the cache hash key
    We want to enable a auto seed generator to get different default seed for `_create_chat_completion_inner_function`
    if seed is not given.
    So the cache will only not miss you ask the same question on same round.
    """
    init_chat_cache_seed: int = 42

    # Chat configs
    openai_api_key: str = ""  # TODO: simplify the key design.
    chat_openai_api_key: str | None = None
    chat_openai_base_url: str | None = None  #
    chat_azure_api_base: str = ""
    chat_azure_api_version: str = ""
    chat_max_tokens: int | None = None
    chat_temperature: float = 0.5
    chat_stream: bool = True
    chat_seed: int | None = None
    chat_frequency_penalty: float = 0.0
    chat_presence_penalty: float = 0.0
    chat_token_limit: int = (
        100000  # 100000 is the maximum limit of gpt4, which might increase in the future version of gpt
    )
    default_system_prompt: str = "You are an AI assistant who helps to answer user's questions."
    system_prompt_role: str = "system"
    """Some models (like o1) do not support the 'system' role.
    Therefore, we make the system_prompt_role customizable to ensure successful calls."""

    # Embedding configs
    embedding_openai_api_key: str = ""
    embedding_openai_base_url: str = ""
    embedding_azure_api_base: str = ""
    embedding_azure_api_version: str = ""
    embedding_max_str_num: int = 50

    # offline llama2 related config
    use_llama2: bool = False
    llama2_ckpt_dir: str = "Llama-2-7b-chat"
    llama2_tokenizer_path: str = "Llama-2-7b-chat/tokenizer.model"
    llams2_max_batch_size: int = 8

    # server served endpoints
    use_gcr_endpoint: bool = False
    gcr_endpoint_type: str = "llama2_70b"  # or "llama3_70b", "phi2", "phi3_4k", "phi3_128k"

    llama2_70b_endpoint: str = ""
    llama2_70b_endpoint_key: str = ""
    llama2_70b_endpoint_deployment: str = ""

    llama3_70b_endpoint: str = ""
    llama3_70b_endpoint_key: str = ""
    llama3_70b_endpoint_deployment: str = ""

    phi2_endpoint: str = ""
    phi2_endpoint_key: str = ""
    phi2_endpoint_deployment: str = ""

    phi3_4k_endpoint: str = ""
    phi3_4k_endpoint_key: str = ""
    phi3_4k_endpoint_deployment: str = ""

    phi3_128k_endpoint: str = ""
    phi3_128k_endpoint_key: str = ""
    phi3_128k_endpoint_deployment: str = ""

    gcr_endpoint_temperature: float = 0.7
    gcr_endpoint_top_p: float = 0.9
    gcr_endpoint_do_sample: bool = False
    gcr_endpoint_max_token: int = 100

    chat_use_azure_deepseek: bool = False
    chat_azure_deepseek_endpoint: str = ""
    chat_azure_deepseek_key: str = ""

    chat_model_map: dict[str, dict[str, str]] = {}


LLM_SETTINGS = LLMSettings()
