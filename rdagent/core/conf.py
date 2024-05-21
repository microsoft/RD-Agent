# TODO: use pydantic for other modules in Qlib
# from pydantic_settings import BaseSettings
import os
from typing import Union

from dotenv import load_dotenv

# make sure that env variable is loaded while calling Config()
load_dotenv(verbose=True, override=True)

from pydantic_settings import BaseSettings


class FincoSettings(BaseSettings):
    use_azure: bool = True
    max_retry: int = 10
    retry_wait_seconds: int = 1
    continuous_mode: bool = False
    dump_chat_cache: bool = False
    use_chat_cache: bool = False
    dump_embedding_cache: bool = False
    use_embedding_cache: bool = False
    prompt_cache_path: str = os.getcwd() + "/prompt_cache.db"
    session_cache_folder_location: str = os.getcwd() + "/session_cache_folder/"
    max_past_message_include: int = 10

    log_llm_chat_content: bool = True

    # Chat configs
    chat_openai_api_key: str = ""
    chat_azure_api_base: str = ""
    chat_azure_api_version: str = ""
    chat_model: str = ""
    chat_max_tokens: int = 3000
    chat_temperature: float = 0.5
    chat_stream: bool = True
    chat_seed: Union[int, None] = None
    chat_frequency_penalty: float = 0.0
    chat_presence_penalty: float = 0.0

    default_system_prompt: str = "You are an AI assistant who helps to answer user's questions about finance."

    # Embedding configs
    embedding_openai_api_key: str = ""
    embedding_azure_api_base: str = ""
    embedding_azure_api_version: str = ""
    embedding_model: str = ""
