"""
Adapter tools for pydantic-ai
"""
import os
from rdagent.oai.llm_conf import LLM_SETTINGS
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider
from litellm.utils import get_llm_provider

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
    assert LLM_SETTINGS.backend.endswith("LiteLLMAPIBackend"), "Only LiteLLMAPIBackend is supported"

    _, custom_llm_provider, _, _ = get_llm_provider(LLM_SETTINGS.chat_model)
    assert custom_llm_provider in PROVIDER_TO_ENV_MAP, f"Provider {custom_llm_provider} not supported. Please add it into `PROVIDER_TO_ENV_MAP`"
    prefix = PROVIDER_TO_ENV_MAP[custom_llm_provider]
    api_key = os.getenv(f"{prefix}_API_KEY", None)
    api_base = os.getenv(f"{prefix}_API_BASE", None)
    return OpenAIChatModel(LLM_SETTINGS.chat_model, provider=LiteLLMProvider(api_base=api_key, api_key=api_base))
