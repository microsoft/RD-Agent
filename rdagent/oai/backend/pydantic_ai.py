"""
Adapter tools for pydantic-ai
"""

import os

from litellm.utils import get_llm_provider
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.litellm import LiteLLMProvider

from rdagent.oai.backend.litellm import LiteLLMAPIBackend
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend

# NOTE:
# LiteLLM's code is not well orgnized.
# we can't reuse any component to map the provider to the env name
# So we have to hardcode on here.
PROVIDER_TO_ENV_MAP = {
    "openai": "OPENAI",
    "azure_ai": "AZURE_AI",
    "azure": "AZURE",
    "litellm_proxy": "LITELLM_PROXY",
}


def get_agent_model() -> OpenAIChatModel:
    """
    Converting LiteLLM to a pydantic-ai model. So you can use like this

    .. code-block:: python

        from rdagent.oai.backend.pydantic_ai import get_agent_model
        model = get_agent_model()
        agent = Agent(model)

    """
    backend = APIBackend()
    assert isinstance(backend, LiteLLMAPIBackend), "Only LiteLLMAPIBackend is supported"

    compl_kwargs = backend.get_complete_kwargs()

    selected_model = compl_kwargs["model"]

    _, custom_llm_provider, _, _ = get_llm_provider(selected_model)
    assert (
        custom_llm_provider in PROVIDER_TO_ENV_MAP
    ), f"Provider {custom_llm_provider} not supported. Please add it into `PROVIDER_TO_ENV_MAP`"
    prefix = PROVIDER_TO_ENV_MAP[custom_llm_provider]
    api_key = os.getenv(f"{prefix}_API_KEY", None)
    api_base = os.getenv(f"{prefix}_API_BASE", None)

    kwargs = {
        "openai_reasoning_effort": compl_kwargs.get("reasoning_effort"),
        "max_tokens": compl_kwargs.get("max_tokens"),
        "temperature": compl_kwargs.get("temperature"),
    }
    if compl_kwargs.get("max_tokens") is None:
        kwargs["max_tokens"] = LLM_SETTINGS.chat_max_tokens
    settings = OpenAIChatModelSettings(**kwargs)
    return OpenAIChatModel(
        selected_model, provider=LiteLLMProvider(api_base=api_base, api_key=api_key), settings=settings
    )
