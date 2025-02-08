import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from litellm import acompletion, completion
from litellm import encode as encode_litellm
from litellm import token_counter

from rdagent.core.conf import ExtendedBaseSettings
from rdagent.core.utils import LLM_CACHE_SEED_GEN, SingletonBaseClass, import_class
from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import APIBackend
from rdagent.oai.llm_conf import LLM_SETTINGS


class LiteLLMSettings(ExtendedBaseSettings):

    class Config:
        env_prefix = "LITELLM_"
        """Use `LITELLM_` as prefix for environment variables"""

    # LiteLLM backend related config
    chat_model: str = "openai/gpt-4o"
    # LiteLLM embedding related config
    embedding_model: str = "openai/text-embedding-3-small"


LITELLM_SETTINGS = LiteLLMSettings()


class LiteLLMAPIBackend(APIBackend):
    """LiteLLM implementation of APIBackend interface"""

    def __init__(self, litellm_model_name: str = "", litellm_api_key: str = "", *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning("LiteLLM backend does not support any additional arguments")

    def build_chat_session(
        self, conversation_id: Optional[str] = None, session_system_prompt: Optional[str] = None
    ) -> Any:
        """Create a new chat session using LiteLLM"""
        # return {
        #     "conversation_id": conversation_id or str(uuid.uuid4()),
        #     "system_prompt": session_system_prompt,
        #     "messages": []
        # }
        raise NotImplementedError("LiteLLM backend does not support chat session creation")
        # TODO: Implement the chat session creation logic , with ChatSession class

    def build_messages_and_create_chat_completion(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        former_messages: Optional[List[Any]] = None,
        chat_cache_prefix: str = "",
        shrink_multiple_break: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Build messages and get LiteLLM chat completion"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if former_messages:
            messages.extend(former_messages)

        messages.append({"role": "user", "content": user_prompt})
        model_name = LITELLM_SETTINGS.chat_model
        # Call LiteLLM completion
        response = completion(
            model=model_name,
            messages=messages,
            stream=kwargs.get("stream", False),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000),
            **kwargs,
        )
        logger.info(
            f"{LogColors.GREEN}Using chat model{LogColors.END} {model_name}",
            tag="debug_llm",
        )

        if system_prompt:
            logger.info(f"{LogColors.RED}system:{LogColors.END} {system_prompt}", tag="debug_llm")
        if former_messages:
            for message in former_messages:
                logger.info(f"{LogColors.CYAN}{message['role']}:{LogColors.END} {message['content']}", tag="debug_llm")
        else:
            logger.info(
                f"{LogColors.RED}user:{LogColors.END} {user_prompt}\n{LogColors.BLUE}resp(next row):\n{LogColors.END} {response.choices[0].message.content}",
                tag="debug_llm",
            )

        return str(response.choices[0].message.content)

    def create_embedding(self, input_content: str | list[str], *args: Any, **kwargs: Any) -> list[Any] | Any:
        """Create embeddings using LiteLLM"""
        from litellm import embedding

        single_input = False
        if isinstance(input_content, str):
            input_content = [input_content]
            single_input = True
        response_list = []
        for input_content_iter in input_content:
            model_name = LITELLM_SETTINGS.embedding_model or "azure/text-embedding-3-small"
            logger.info(f"{LogColors.GREEN}Using emb model{LogColors.END} {model_name}", tag="debug_litellm_emb")
            logger.info(f"Creating embedding for: {input_content_iter}", tag="debug_litellm_emb")
            if not isinstance(input_content_iter, str):
                raise ValueError("Input content must be a string")
            response = embedding(
                model=model_name,
                input=input_content_iter,
                **kwargs,
            )
            response_list.append(response.data[0]["embedding"])
        if single_input:
            return response_list[0]
        return response_list

    def build_messages_and_calculate_token(
        self,
        user_prompt: str,
        system_prompt: Optional[str],
        former_messages: Optional[List[Dict[str, Any]]] = None,
        shrink_multiple_break: bool = False,
    ) -> int:
        """Build messages and calculate their token count using LiteLLM"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if former_messages:
            messages.extend(former_messages)

        messages.append({"role": "user", "content": user_prompt})

        num_tokens = token_counter(
            model=LITELLM_SETTINGS.chat_model,
            messages=messages,
        )
        logger.info(f"{LogColors.CYAN}Token count: {LogColors.END} {num_tokens}", tag="debug_litellm_token")
        return num_tokens
