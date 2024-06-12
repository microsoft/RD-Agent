from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# TODO: use pydantic for other modules in Qlib
# from pydantic_settings import BaseSettings

# make sure that env variable is loaded while calling Config()
load_dotenv(verbose=True, override=True)


class FincoSettings(BaseSettings):
    use_azure: bool = True
    use_azure_token_provider: bool = False
    max_retry: int = 10
    retry_wait_seconds: int = 1
    continuous_mode: bool = False
    dump_chat_cache: bool = False
    use_chat_cache: bool = False
    dump_embedding_cache: bool = False
    use_embedding_cache: bool = False
    workspace: str = "./finco_workspace"
    prompt_cache_path: str = str(Path.cwd() / "prompt_cache.db")
    session_cache_folder_location: str = str(Path.cwd() / "session_cache_folder/")
    max_past_message_include: int = 10

    use_vector_only: bool = False
    log_llm_chat_content: bool = True

    # Chat configs
    chat_openai_api_key: str = ""
    chat_azure_api_base: str = ""
    chat_azure_api_version: str = ""
    chat_model: str = ""
    chat_max_tokens: int = 3000
    chat_temperature: float = 0.5
    chat_stream: bool = True
    chat_seed: int | None = None
    chat_frequency_penalty: float = 0.0
    chat_presence_penalty: float = 0.0

    default_system_prompt: str = "You are an AI assistant who helps to answer user's questions about finance."

    # Embedding configs
    embedding_openai_api_key: str = ""
    embedding_azure_api_base: str = ""
    embedding_azure_api_version: str = ""
    embedding_model: str = ""

    # llama2 related config
    use_llama2: bool = False
    llama2_ckpt_dir: str = "Llama-2-7b-chat"
    llama2_tokenizer_path: str = "Llama-2-7b-chat/tokenizer.model"
    llams2_max_batch_size: int = 8

    # finco v2 configs
    azure_document_intelligence_key: str = ""
    azure_document_intelligence_endpoint: str = ""

    # fincov2 llama2 endpoint
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

    # factor extraction conf
    max_input_duplicate_factor_group: int = 600
    max_output_duplicate_factor_group: int = 20
