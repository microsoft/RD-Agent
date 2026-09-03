"""Unit tests for the upfront ``<provider>/<model>`` validation added in
response to #1016.

These tests do not hit any live LLM endpoint; they only exercise the
``get_llm_provider`` resolution that LiteLLM exposes purely client-side.
"""

import unittest

import pytest

# get_llm_provider is the same helper the validation function uses; this
# avoids tripping over the lazy environment requirements of constructing a
# full LiteLLMAPIBackend instance.
try:
    from litellm.exceptions import BadRequestError
    from litellm.utils import get_llm_provider
except ImportError:  # pragma: no cover - litellm is a hard dependency of RD-Agent
    pytest.skip("litellm is not installed; skipping provider validation tests", allow_module_level=True)

from rdagent.oai.backend.litellm import _validate_litellm_model_provider


class TestValidateLiteLLMModelProvider(unittest.TestCase):
    def test_default_chat_model_passes(self) -> None:
        # gpt-4-turbo is the default; LiteLLM resolves bare OpenAI names.
        _validate_litellm_model_provider("gpt-4-turbo", "CHAT_MODEL")

    def test_default_embedding_model_passes(self) -> None:
        _validate_litellm_model_provider("text-embedding-3-small", "EMBEDDING_MODEL")

    def test_provider_prefixed_chat_models_pass(self) -> None:
        for model in [
            "openai/gpt-4o",
            "deepseek/deepseek-chat",
            "ollama/qwen2:7b",
            "azure/my-deployment",
        ]:
            with self.subTest(model=model):
                _validate_litellm_model_provider(model, "CHAT_MODEL")

    def test_bare_unknown_chat_model_raises_actionable_error(self) -> None:
        # This is the exact failure mode from #1016: a bare model name that
        # LiteLLM cannot map to a known provider.
        with pytest.raises(RuntimeError) as exc_info:
            _validate_litellm_model_provider("definitely-not-a-real-model-xyz", "CHAT_MODEL")
        message = str(exc_info.value)
        # Error must surface the offending env var name, the bad value, and
        # at least one corrected example, so the user can fix it without
        # having to grep the source.
        self.assertIn("CHAT_MODEL", message)
        self.assertIn("definitely-not-a-real-model-xyz", message)
        self.assertIn("<provider>/<model>", message)
        # The chained cause should still be the original LiteLLM exception so
        # advanced users can inspect it.
        self.assertIsInstance(exc_info.value.__cause__, BadRequestError)

    def test_bare_unknown_embedding_model_raises_actionable_error(self) -> None:
        with pytest.raises(RuntimeError) as exc_info:
            _validate_litellm_model_provider("definitely-not-a-real-emb-xyz", "EMBEDDING_MODEL")
        self.assertIn("EMBEDDING_MODEL", str(exc_info.value))


if __name__ == "__main__":
    unittest.main()
