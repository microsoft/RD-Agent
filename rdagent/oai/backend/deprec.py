from __future__ import annotations

import inspect
import json
import os
import random
import re
import sqlite3
import ssl
import time
import urllib.request
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Type, Union, cast

import numpy as np
import openai
import tiktoken
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from rdagent.core.utils import LLM_CACHE_SEED_GEN, SingletonBaseClass, import_class
from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.utils import md5_hash

DEFAULT_QLIB_DOT_PATH = Path("./")

from rdagent.oai.backend.base import APIBackend

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
except ImportError:
    logger.warning("azure.identity is not installed.")

try:
    import openai
except ImportError:
    logger.warning("openai is not installed.")

try:
    from llama import Llama
except ImportError:
    if LLM_SETTINGS.use_llama2:
        logger.warning("llama is not installed.")

try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import (
        AssistantMessage,
        ChatRequestMessage,
        SystemMessage,
        UserMessage,
    )
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    if LLM_SETTINGS.chat_use_azure_deepseek:
        logger.warning("azure.ai.inference or azure.core.credentials is not installed.")


class ConvManager:
    """
    This is a conversation manager of LLM
    It is for convenience of exporting conversation for debugging.
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_QLIB_DOT_PATH / "llm_conv",
        recent_n: int = 10,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.recent_n = recent_n

    def _rotate_files(self) -> None:
        pairs = []
        for f in self.path.glob("*.json"):
            m = re.match(r"(\d+).json", f.name)
            if m is not None:
                n = int(m.group(1))
                pairs.append((n, f))
        pairs.sort(key=lambda x: x[0])
        for n, f in pairs[: self.recent_n][::-1]:
            if (self.path / f"{n+1}.json").exists():
                (self.path / f"{n+1}.json").unlink()
            f.rename(self.path / f"{n+1}.json")

    def append(self, conv: tuple[list, str]) -> None:
        self._rotate_files()
        with (self.path / "0.json").open("w") as file:
            json.dump(conv, file)
        # TODO: reseve line breaks to make it more convient to edit file directly.


class DeprecBackend(APIBackend):
    """
    This is a unified interface for different backends.

    (xiao) thinks integrate all kinds of API in a single class is not a good design.
    So we should split them into different classes in `oai/backends/` in the future.
    """

    # FIXME: (xiao) We should avoid using self.xxxx.
    # Instead, we can use LLM_SETTINGS directly. If it's difficult to support different backend settings, we can split them into multiple BaseSettings.
    def __init__(  # noqa: C901, PLR0912, PLR0915
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if LLM_SETTINGS.use_llama2:
            self.generator = Llama.build(
                ckpt_dir=LLM_SETTINGS.llama2_ckpt_dir,
                tokenizer_path=LLM_SETTINGS.llama2_tokenizer_path,
                max_seq_len=LLM_SETTINGS.chat_max_tokens,
                max_batch_size=LLM_SETTINGS.llams2_max_batch_size,
            )
            self.encoder = None
        elif LLM_SETTINGS.use_gcr_endpoint:
            gcr_endpoint_type = LLM_SETTINGS.gcr_endpoint_type
            if gcr_endpoint_type == "llama2_70b":
                self.gcr_endpoint_key = LLM_SETTINGS.llama2_70b_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.llama2_70b_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.llama2_70b_endpoint
            elif gcr_endpoint_type == "llama3_70b":
                self.gcr_endpoint_key = LLM_SETTINGS.llama3_70b_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.llama3_70b_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.llama3_70b_endpoint
            elif gcr_endpoint_type == "phi2":
                self.gcr_endpoint_key = LLM_SETTINGS.phi2_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.phi2_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.phi2_endpoint
            elif gcr_endpoint_type == "phi3_4k":
                self.gcr_endpoint_key = LLM_SETTINGS.phi3_4k_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.phi3_4k_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.phi3_4k_endpoint
            elif gcr_endpoint_type == "phi3_128k":
                self.gcr_endpoint_key = LLM_SETTINGS.phi3_128k_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.phi3_128k_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.phi3_128k_endpoint
            else:
                error_message = f"Invalid gcr_endpoint_type: {gcr_endpoint_type}"
                raise ValueError(error_message)
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + self.gcr_endpoint_key),
            }
            self.gcr_endpoint_temperature = LLM_SETTINGS.gcr_endpoint_temperature
            self.gcr_endpoint_top_p = LLM_SETTINGS.gcr_endpoint_top_p
            self.gcr_endpoint_do_sample = LLM_SETTINGS.gcr_endpoint_do_sample
            self.gcr_endpoint_max_token = LLM_SETTINGS.gcr_endpoint_max_token
            if not os.environ.get("PYTHONHTTPSVERIFY", "") and hasattr(ssl, "_create_unverified_context"):
                ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
            self.chat_model_map = LLM_SETTINGS.chat_model_map
            self.chat_model = LLM_SETTINGS.chat_model
            self.encoder = None
        elif LLM_SETTINGS.chat_use_azure_deepseek:
            self.client = ChatCompletionsClient(
                endpoint=LLM_SETTINGS.chat_azure_deepseek_endpoint,
                credential=AzureKeyCredential(LLM_SETTINGS.chat_azure_deepseek_key),
            )
            self.chat_model_map = LLM_SETTINGS.chat_model_map
            self.encoder = None
            self.chat_model = "deepseek-R1"
            self.chat_stream = LLM_SETTINGS.chat_stream
        else:
            self.chat_use_azure = LLM_SETTINGS.chat_use_azure or LLM_SETTINGS.use_azure
            self.embedding_use_azure = LLM_SETTINGS.embedding_use_azure or LLM_SETTINGS.use_azure
            self.chat_use_azure_token_provider = LLM_SETTINGS.chat_use_azure_token_provider
            self.embedding_use_azure_token_provider = LLM_SETTINGS.embedding_use_azure_token_provider
            self.managed_identity_client_id = LLM_SETTINGS.managed_identity_client_id

            # Priority: chat_api_key/embedding_api_key > openai_api_key > os.environ.get("OPENAI_API_KEY")
            # TODO: Simplify the key design. Consider Pandatic's field alias & priority.
            self.chat_api_key = (
                LLM_SETTINGS.chat_openai_api_key or LLM_SETTINGS.openai_api_key or os.environ.get("OPENAI_API_KEY")
            )
            self.embedding_api_key = (
                LLM_SETTINGS.embedding_openai_api_key or LLM_SETTINGS.openai_api_key or os.environ.get("OPENAI_API_KEY")
            )

            self.chat_model = LLM_SETTINGS.chat_model
            self.chat_model_map = LLM_SETTINGS.chat_model_map
            self.encoder = self._get_encoder()
            self.chat_openai_base_url = LLM_SETTINGS.chat_openai_base_url
            self.embedding_openai_base_url = LLM_SETTINGS.embedding_openai_base_url
            self.chat_api_base = LLM_SETTINGS.chat_azure_api_base
            self.chat_api_version = LLM_SETTINGS.chat_azure_api_version
            self.chat_stream = LLM_SETTINGS.chat_stream
            self.chat_seed = LLM_SETTINGS.chat_seed

            self.embedding_model = LLM_SETTINGS.embedding_model
            self.embedding_api_base = LLM_SETTINGS.embedding_azure_api_base
            self.embedding_api_version = LLM_SETTINGS.embedding_azure_api_version

            if (self.chat_use_azure or self.embedding_use_azure) and (
                self.chat_use_azure_token_provider or self.embedding_use_azure_token_provider
            ):
                dac_kwargs = {}
                if self.managed_identity_client_id is not None:
                    dac_kwargs["managed_identity_client_id"] = self.managed_identity_client_id
                credential = DefaultAzureCredential(**dac_kwargs)
                token_provider = get_bearer_token_provider(
                    credential,
                    "https://cognitiveservices.azure.com/.default",
                )
            self.chat_client: openai.OpenAI = (
                openai.AzureOpenAI(
                    azure_ad_token_provider=token_provider if self.chat_use_azure_token_provider else None,
                    api_key=self.chat_api_key if not self.chat_use_azure_token_provider else None,
                    api_version=self.chat_api_version,
                    azure_endpoint=self.chat_api_base,
                )
                if self.chat_use_azure
                else openai.OpenAI(api_key=self.chat_api_key, base_url=self.chat_openai_base_url)
            )

            self.embedding_client: openai.OpenAI = (
                openai.AzureOpenAI(
                    azure_ad_token_provider=token_provider if self.embedding_use_azure_token_provider else None,
                    api_key=self.embedding_api_key if not self.embedding_use_azure_token_provider else None,
                    api_version=self.embedding_api_version,
                    azure_endpoint=self.embedding_api_base,
                )
                if self.embedding_use_azure
                else openai.OpenAI(api_key=self.embedding_api_key, base_url=self.embedding_openai_base_url)
            )

        # transfer the config to the class if the config is not supposed to change during the runtime
        self.use_llama2 = LLM_SETTINGS.use_llama2
        self.use_gcr_endpoint = LLM_SETTINGS.use_gcr_endpoint
        self.chat_use_azure_deepseek = LLM_SETTINGS.chat_use_azure_deepseek

    def _get_encoder(self) -> tiktoken.Encoding:
        """
        tiktoken.encoding_for_model(self.chat_model) does not cover all cases it should consider.

        This function attempts to handle several edge cases.
        """

        # 1) cases
        def _azure_patch(model: str) -> str:
            """
            When using Azure API, self.chat_model is the deployment name that can be any string.
            For example, it may be `gpt-4o_2024-08-06`. But tiktoken.encoding_for_model can't handle this.
            """
            return model.replace("_", "-")

        model = self.chat_model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Failed to get encoder. Trying to patch the model name")
            for patch_func in [_azure_patch]:
                try:
                    encoding = tiktoken.encoding_for_model(patch_func(model))
                except KeyError:
                    logger.error(f"Failed to get encoder even after patching with {patch_func.__name__}")
                    raise
        return encoding

    def supports_response_schema(self) -> bool:
        """
        Check if the backend supports function calling.
        Currently, deprec backend does not support function calling so it returns False. #FIXME: maybe a mapping to the backend class is needed.
        """
        return False

    def _create_embedding_inner_function(  # type: ignore[no-untyped-def]
        self, input_content_list: list[str], *args, **kwargs
    ) -> list[list[float]]:  # noqa: ARG002
        content_to_embedding_dict = {}
        for sliced_filtered_input_content_list in [
            input_content_list[i : i + LLM_SETTINGS.embedding_max_str_num]
            for i in range(0, len(input_content_list), LLM_SETTINGS.embedding_max_str_num)
        ]:
            if self.embedding_use_azure:
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=sliced_filtered_input_content_list,
                )
            else:
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=sliced_filtered_input_content_list,
                )
            for index, data in enumerate(response.data):
                content_to_embedding_dict[sliced_filtered_input_content_list[index]] = data.embedding

        return [content_to_embedding_dict[content] for content in input_content_list]

    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def] # noqa: C901, PLR0912, PLR0915
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        add_json_in_prompt: bool = False,
        *args,
        **kwargs,
    ) -> tuple[str, str | None]:
        """
        seed : Optional[int]
            When retrying with cache enabled, it will keep returning the same results.
            To make retries useful, we need to enable a seed.
            This seed is different from `self.chat_seed` for GPT. It is for the local cache mechanism enabled by RD-Agent locally.
        """

        # TODO: we can add this function back to avoid so much `self.cfg.log_llm_chat_content`
        if LLM_SETTINGS.log_llm_chat_content:
            logger.info(self._build_log_messages(messages), tag="llm_messages")
        # TODO: fail to use loguru adaptor due to stream response

        model = LLM_SETTINGS.chat_model
        temperature = LLM_SETTINGS.chat_temperature
        max_tokens = LLM_SETTINGS.chat_max_tokens
        frequency_penalty = LLM_SETTINGS.chat_frequency_penalty
        presence_penalty = LLM_SETTINGS.chat_presence_penalty

        if self.chat_model_map:
            for t, mc in self.chat_model_map.items():
                if t in logger._tag:
                    model = mc.get("model", model)
                    temperature = float(mc.get("temperature", temperature))
                    if "max_tokens" in mc:
                        max_tokens = int(mc["max_tokens"])
                    break

        finish_reason = None
        if self.use_llama2:
            response = self.generator.chat_completion(
                messages,
                max_gen_len=max_tokens,
                temperature=temperature,
            )
            resp = response[0]["generation"]["content"]
            if LLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.CYAN}Response:{resp}{LogColors.END}", tag="llm_messages")
        elif self.use_gcr_endpoint:
            body = str.encode(
                json.dumps(
                    {
                        "input_data": {
                            "input_string": messages,
                            "parameters": {
                                "temperature": self.gcr_endpoint_temperature,
                                "top_p": self.gcr_endpoint_top_p,
                                "max_new_tokens": self.gcr_endpoint_max_token,
                            },
                        },
                    },
                ),
            )

            req = urllib.request.Request(self.gcr_endpoint, body, self.headers)  # noqa: S310
            response = urllib.request.urlopen(req)  # noqa: S310
            resp = json.loads(response.read().decode())["output"]
            if LLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.CYAN}Response:{resp}{LogColors.END}", tag="llm_messages")
        elif self.chat_use_azure_deepseek:
            azure_style_message: list[ChatRequestMessage] = []
            for message in messages:
                if message["role"] == "system":
                    azure_style_message.append(SystemMessage(content=message["content"]))
                elif message["role"] == "user":
                    azure_style_message.append(UserMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    azure_style_message.append(AssistantMessage(content=message["content"]))

            response = self.client.complete(
                messages=azure_style_message,
                stream=self.chat_stream,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            if self.chat_stream:
                resp = ""
                # TODO: with logger.config(stream=self.chat_stream): and add a `stream_start` flag to add timestamp for first message.
                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info(f"{LogColors.CYAN}Response:{LogColors.END}", tag="llm_messages")

                for chunk in response:
                    content = (
                        chunk.choices[0].delta.content
                        if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None
                        else ""
                    )
                    if LLM_SETTINGS.log_llm_chat_content:
                        logger.info(LogColors.CYAN + content + LogColors.END, raw=True, tag="llm_messages")
                    resp += content
                    if len(chunk.choices) > 0 and chunk.choices[0].finish_reason is not None:
                        finish_reason = chunk.choices[0].finish_reason
            else:
                response = cast(ChatCompletion, response)
                resp = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info(f"{LogColors.CYAN}Response:{resp}{LogColors.END}", tag="llm_messages")
            match = re.search(r"<think>(.*?)</think>(.*)", resp, re.DOTALL)
            think_part, resp = match.groups() if match else ("", resp)
            if LLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.CYAN}Think:{think_part}{LogColors.END}", tag="llm_messages")
                logger.info(f"{LogColors.CYAN}Response:{resp}{LogColors.END}", tag="llm_messages")
        else:
            call_kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=self.chat_stream,
                seed=self.chat_seed,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

            # FIX what if the model does not support response_schema
            if response_format == {"type": "json_object"} and add_json_in_prompt:
                for message in messages[::-1]:
                    message["content"] = message["content"] + "\nPlease respond in json format."
                    if message["role"] == LLM_SETTINGS.system_prompt_role:
                        # NOTE: assumption: systemprompt is always the first message
                        break
                call_kwargs["response_format"] = {"type": "json_object"}
            response = self.chat_client.chat.completions.create(**call_kwargs)

            if self.chat_stream:
                resp = ""
                # TODO: with logger.config(stream=self.chat_stream): and add a `stream_start` flag to add timestamp for first message.
                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info(f"{LogColors.CYAN}Response:{LogColors.END}", tag="llm_messages")

                for chunk in response:
                    content = (
                        chunk.choices[0].delta.content
                        if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None
                        else ""
                    )
                    if LLM_SETTINGS.log_llm_chat_content:
                        logger.info(LogColors.CYAN + content + LogColors.END, raw=True, tag="llm_messages")
                    resp += content
                    if len(chunk.choices) > 0 and chunk.choices[0].finish_reason is not None:
                        finish_reason = chunk.choices[0].finish_reason

                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info("\n", raw=True, tag="llm_messages")

            else:
                resp = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info(f"{LogColors.CYAN}Response:{resp}{LogColors.END}", tag="llm_messages")
                    logger.info(
                        json.dumps(
                            {
                                "total_tokens": response.usage.total_tokens,
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "model": model,
                            }
                        ),
                        tag="llm_messages",
                    )
        return resp, finish_reason

    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        if self.chat_use_azure_deepseek:
            return 0
        if self.encoder is None:
            raise ValueError("Encoder is not initialized.")
        if self.use_llama2 or self.use_gcr_endpoint:
            logger.warning("num_tokens_from_messages() is not implemented for model llama2.")
            return 0  # TODO implement this function for llama2

        if "gpt4" in self.chat_model or "gpt-4" in self.chat_model:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            tokens_per_message = 4  # every message follows <start>{role/name}\n{content}<end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encoder.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <start>assistant<message>
        return num_tokens
