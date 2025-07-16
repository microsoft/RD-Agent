import os

import fire
import litellm
from litellm import completion, embedding
from litellm.utils import ModelResponse

from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.litellm import LITELLM_SETTINGS


def test_chat(chat_model, chat_api_key, chat_api_base):
    logger.info(f"🧪 Testing chat model: {chat_model}")
    try:
        if chat_api_base is None:
            response: ModelResponse = completion(
                model=chat_model,
                api_key=chat_api_key,
                messages=[
                    {"role": "user", "content": "Hello!"},
                ],
            )
        else:
            response: ModelResponse = completion(
                model=chat_model,
                api_key=chat_api_key,
                api_base=chat_api_base,
                messages=[
                    {"role": "user", "content": "Hello!"},
                ],
            )
        logger.info(f"✅ Chat test passed.")
        return True
    except Exception as e:
        logger.error(f"❌ Chat test failed: {e}")
        return False


def test_embedding(embedding_model, embedding_api_key, embedding_api_base):
    logger.info(f"🧪 Testing embedding model: {embedding_model}")
    try:
        response = embedding(
            model=embedding_model,
            api_key=embedding_api_key,
            api_base=embedding_api_base,
            input="Hello world!",
        )
        logger.info("✅ Embedding test passed.")
        return True
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {e}")
        return False


def main():
    if "BACKEND" not in os.environ:
        logger.warning(
            f"We did not find BACKEND in your configuration, please add it to your .env file. "
            f"You can run a command like this: `dotenv set BACKEND rdagent.oai.backend.LiteLLMAPIBackend`"
        )

    if "DEEPSEEK_API_KEY" in os.environ:
        chat_api_key = os.getenv("DEEPSEEK_API_KEY")
        chat_model = os.getenv("CHAT_MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        embedding_api_key = os.getenv("LITELLM_PROXY_API_KEY")
        embedding_api_base = os.getenv("LITELLM_PROXY_API_BASE")
        if "DEEPSEEK_API_BASE" in os.environ:
            chat_api_base = os.getenv("DEEPSEEK_API_BASE")
        elif "OPENAI_API_BASE" in os.environ:
            chat_api_base = os.getenv("OPENAI_API_BASE")
        else:
            chat_api_base = None
    elif "OPENAI_API_KEY" in os.environ:
        chat_api_key = os.getenv("OPENAI_API_KEY")
        chat_api_base = os.getenv("OPENAI_API_BASE")
        chat_model = os.getenv("CHAT_MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        embedding_api_key = chat_api_key
        embedding_api_base = chat_api_base
    else:
        print("No valid configuration was found, please check your .env file.")

    print("🚀 Starting test...\n")
    result_embedding = test_embedding(
        embedding_model=embedding_model, embedding_api_key=embedding_api_key, embedding_api_base=embedding_api_base
    )
    result_chat = test_chat(chat_model=chat_model, chat_api_key=chat_api_key, chat_api_base=chat_api_base)

    if result_chat and result_embedding:
        print("\n✅ All tests completed.")
    else:
        print(" One or more tests failed. Please check credentials or model support.")
