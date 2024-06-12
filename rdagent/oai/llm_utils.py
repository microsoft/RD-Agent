from __future__ import annotations

import datetime
import hashlib
import json
import multiprocessing
import os
import re
import sqlite3
import ssl
import time
import urllib.request
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from rdagent.core.conf import FincoSettings as Config
from rdagent.core.log import FinCoLog, LogColors
from rdagent.core.utils import SingletonBaseClass

DEFAULT_QLIB_DOT_PATH = Path("./")


def md5_hash(input_string: str) -> str:
    hash_md5 = hashlib.md5(usedforsecurity=False)
    input_bytes = input_string.encode("utf-8")
    hash_md5.update(input_bytes)
    return hash_md5.hexdigest()


try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
except ImportError:
    FinCoLog().warning("azure.identity is not installed.")

try:
    import openai
except ImportError:
    FinCoLog().warning("openai is not installed.")

try:
    from llama import Llama
except ImportError:
    FinCoLog().warning("llama is not installed.")


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


class SQliteLazyCache(SingletonBaseClass):
    def __init__(self, cache_location: str) -> None:
        super().__init__()
        self.cache_location = cache_location
        db_file_exist = Path(cache_location).exists()
        self.conn = sqlite3.connect(cache_location)
        self.c = self.conn.cursor()
        if not db_file_exist:
            self.c.execute(
                """
                CREATE TABLE chat_cache (
                    md5_key TEXT PRIMARY KEY,
                    chat TEXT
                )
                """,
            )
            self.c.execute(
                """
                CREATE TABLE embedding_cache (
                    md5_key TEXT PRIMARY KEY,
                    embedding TEXT
                )
                """,
            )
            self.conn.commit()

    def chat_get(self, key: str) -> str | None:
        md5_key = md5_hash(key)
        self.c.execute("SELECT chat FROM chat_cache WHERE md5_key=?", (md5_key,))
        result = self.c.fetchone()
        if result is None:
            return None
        return result[0]

    def embedding_get(self, key: str) -> list | dict | str | None:
        md5_key = md5_hash(key)
        self.c.execute("SELECT embedding FROM embedding_cache WHERE md5_key=?", (md5_key,))
        result = self.c.fetchone()
        if result is None:
            return None
        return json.loads(result[0])

    def chat_set(self, key: str, value: str) -> None:
        md5_key = md5_hash(key)
        self.c.execute(
            "INSERT OR REPLACE INTO chat_cache (md5_key, chat) VALUES (?, ?)",
            (md5_key, value),
        )
        self.conn.commit()


    def embedding_set(self, content_to_embedding_dict: dict) -> None:
        for key, value in content_to_embedding_dict.items():
            md5_key = md5_hash(key)
            self.c.execute(
                "INSERT OR REPLACE INTO embedding_cache (md5_key, embedding) VALUES (?, ?)",
                (md5_key, json.dumps(value)),
            )
        self.conn.commit()


class SessionChatHistoryCache(SingletonBaseClass):
    def __init__(self) -> None:
        """load all history conversation json file from self.session_cache_location"""
        self.cfg = Config()
        self.session_cache_location = Path(self.cfg.session_cache_folder_location)
        self.cache = {}
        if not self.session_cache_location.exists():
            FinCoLog.warning(f"Directory {self.session_cache_location} does not exist.")
            self.session_cache_location.mkdir(parents=True, exist_ok=True)
        json_files = [f for f in self.session_cache_location.iterdir() if f.suffix == ".json"]
        if not json_files:
            FinCoLog.info(f"No JSON files found in {self.session_cache_location}.")
        for file_path in json_files:
            conversation_id = file_path.stem
            with file_path.open("r") as f:
                conversation_content = json.load(f)
                self.cache[conversation_id] = conversation_content["content"]

    def message_get(self, conversation_id: str) -> list[str]:
        return self.cache.get(conversation_id, [])

    def message_set(self, conversation_id: str, message_value: list[str]) -> None:
        self.cache[conversation_id] = message_value
        conversation_path = self.session_cache_location / conversation_id
        conversation_path = conversation_path.with_suffix(".json")
        current_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
        with conversation_path.open("w") as f:
            json.dump({"content": message_value, "last_modified_time": current_time}, f)


class ChatSession:
    def __init__(self, api_backend: Any, conversation_id: str | None = None, system_prompt: str | None = None) -> None:
        self.conversation_id = str(uuid.uuid4()) if conversation_id is None else conversation_id
        self.cfg = Config()
        self.system_prompt = system_prompt if system_prompt is not None else self.cfg.default_system_prompt
        self.api_backend = api_backend

    def build_chat_completion_message(self, user_prompt: str) -> list[dict[str, Any]]:
        history_message = SessionChatHistoryCache().message_get(self.conversation_id)
        messages = history_message
        if not messages:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        return messages

    def build_chat_completion_message_and_calculate_token(self, user_prompt: str) -> Any:
        messages = self.build_chat_completion_message(user_prompt)
        return self.api_backend.calculate_token_from_messages(messages)

    def build_chat_completion(self, user_prompt: str, **kwargs: Any) -> str:
        """
        this function is to build the session messages
        user prompt should always be provided
        """
        messages = self.build_chat_completion_message(user_prompt, **kwargs)

        response = self.api_backend._try_create_chat_completion_or_embedding( # noqa: SLF001
            messages=messages, chat_completion=True, **kwargs,
        )
        messages.append(
            {
                "role": "assistant",
                "content": response,
            },
        )
        SessionChatHistoryCache().message_set(self.conversation_id, messages)
        return response

    def get_conversation_id(self) -> str:
        return self.conversation_id

    def display_history(self) -> None:
        # TODO: Realize a beautiful presentation format for history messages
        pass


class APIBackend:
    def __init__( # noqa: C901, PLR0912, PLR0915
        self,
        *,
        chat_api_key: str | None = None,
        chat_model: str | None = None,
        chat_api_base: str | None = None,
        chat_api_version: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model: str | None = None,
        embedding_api_base: str | None = None,
        embedding_api_version: str | None = None,
        use_chat_cache: bool | None = None,
        dump_chat_cache: bool | None = None,
        use_embedding_cache: bool | None = None,
        dump_embedding_cache: bool | None = None,
    ) -> None:
        self.cfg = Config()
        if self.cfg.use_llama2:
            self.generator = Llama.build(
                ckpt_dir=self.cfg.llama2_ckpt_dir,
                tokenizer_path=self.cfg.llama2_tokenizer_path,
                max_seq_len=self.cfg.max_tokens,
                max_batch_size=self.cfg.llams2_max_batch_size,
            )
            self.encoder = None
        elif self.cfg.use_gcr_endpoint:
            gcr_endpoint_type = self.cfg.gcr_endpoint_type
            if gcr_endpoint_type == "llama2_70b":
                self.gcr_endpoint_key = self.cfg.llama2_70b_endpoint_key
                self.gcr_endpoint_deployment = self.cfg.llama2_70b_endpoint_deployment
                self.gcr_endpoint = self.cfg.llama2_70b_endpoint
            elif gcr_endpoint_type == "llama3_70b":
                self.gcr_endpoint_key = self.cfg.llama3_70b_endpoint_key
                self.gcr_endpoint_deployment = self.cfg.llama3_70b_endpoint_deployment
                self.gcr_endpoint = self.cfg.llama3_70b_endpoint
            elif gcr_endpoint_type == "phi2":
                self.gcr_endpoint_key = self.cfg.phi2_endpoint_key
                self.gcr_endpoint_deployment = self.cfg.phi2_endpoint_deployment
                self.gcr_endpoint = self.cfg.phi2_endpoint
            elif gcr_endpoint_type == "phi3_4k":
                self.gcr_endpoint_key = self.cfg.phi3_4k_endpoint_key
                self.gcr_endpoint_deployment = self.cfg.phi3_4k_endpoint_deployment
                self.gcr_endpoint = self.cfg.phi3_4k_endpoint
            elif gcr_endpoint_type == "phi3_128k":
                self.gcr_endpoint_key = self.cfg.phi3_128k_endpoint_key
                self.gcr_endpoint_deployment = self.cfg.phi3_128k_endpoint_deployment
                self.gcr_endpoint = self.cfg.phi3_128k_endpoint
            else:
                error_message = f"Invalid gcr_endpoint_type: {gcr_endpoint_type}"
                raise ValueError(error_message)
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + self.gcr_endpoint_key),
                "azureml-model-deployment": self.gcr_endpoint_deployment,
            }
            self.gcr_endpoint_temperature = self.cfg.gcr_endpoint_temperature
            self.gcr_endpoint_top_p = self.cfg.gcr_endpoint_top_p
            self.gcr_endpoint_do_sample = self.cfg.gcr_endpoint_do_sample
            self.gcr_endpoint_max_token = self.cfg.gcr_endpoint_max_token
            if not os.environ.get("PYTHONHTTPSVERIFY", "") and hasattr(ssl, "_create_unverified_context"):
                ssl._create_default_https_context = ssl._create_unverified_context # noqa: SLF001
            self.encoder = None
        else:
            self.use_azure = self.cfg.use_azure
            self.use_azure_token_provider = self.cfg.use_azure_token_provider

            self.chat_api_key = self.cfg.chat_openai_api_key if chat_api_key is None else chat_api_key
            self.chat_model = self.cfg.chat_model if chat_model is None else chat_model
            self.encoder = tiktoken.encoding_for_model(self.chat_model)
            self.chat_api_base = self.cfg.chat_azure_api_base if chat_api_base is None else chat_api_base
            self.chat_api_version = self.cfg.chat_azure_api_version if chat_api_version is None else chat_api_version
            self.chat_stream = self.cfg.chat_stream
            self.chat_seed = self.cfg.chat_seed

            self.embedding_api_key = (
                self.cfg.embedding_openai_api_key if embedding_api_key is None else embedding_api_key
            )
            self.embedding_model = self.cfg.embedding_model if embedding_model is None else embedding_model
            self.embedding_api_base = (
                self.cfg.embedding_azure_api_base if embedding_api_base is None else embedding_api_base
            )
            self.embedding_api_version = (
                self.cfg.embedding_azure_api_version if embedding_api_version is None else embedding_api_version
            )

            if self.use_azure:
                if self.use_azure_token_provider:
                    credential = DefaultAzureCredential()
                    token_provider = get_bearer_token_provider(
                        credential, "https://cognitiveservices.azure.com/.default",
                    )
                    self.chat_client = openai.AzureOpenAI(
                        azure_ad_token_provider=token_provider,
                        api_version=self.chat_api_version,
                        azure_endpoint=self.chat_api_base,
                    )
                    self.embedding_client = openai.AzureOpenAI(
                        azure_ad_token_provider=token_provider,
                        api_version=self.embedding_api_version,
                        azure_endpoint=self.embedding_api_base,
                    )
                else:
                    self.chat_client = openai.AzureOpenAI(
                        api_key=self.chat_api_key,
                        api_version=self.chat_api_version,
                        azure_endpoint=self.chat_api_base,
                    )
                    self.embedding_client = openai.AzureOpenAI(
                        api_key=self.embedding_api_key,
                        api_version=self.embedding_api_version,
                        azure_endpoint=self.embedding_api_base,
                    )
            else:
                self.chat_client = openai.OpenAI(api_key=self.chat_api_key)
                self.embedding_client = openai.OpenAI(api_key=self.embedding_api_key)

        self.dump_chat_cache = self.cfg.dump_chat_cache if dump_chat_cache is None else dump_chat_cache
        self.use_chat_cache = self.cfg.use_chat_cache if use_chat_cache is None else use_chat_cache
        self.dump_embedding_cache = (
            self.cfg.dump_embedding_cache if dump_embedding_cache is None else dump_embedding_cache
        )
        self.use_embedding_cache = self.cfg.use_embedding_cache if use_embedding_cache is None else use_embedding_cache
        if self.dump_chat_cache or self.use_chat_cache or self.dump_embedding_cache or self.use_embedding_cache:
            self.cache_file_location = self.cfg.prompt_cache_path
            self.cache = SQliteLazyCache(cache_location=self.cache_file_location)

        # transfer the config to the class if the config is not supposed to change during the runtime
        self.use_llama2 = self.cfg.use_llama2
        self.use_gcr_endpoint = self.cfg.use_gcr_endpoint
        self.retry_wait_seconds = self.cfg.retry_wait_seconds

    def build_chat_session(
        self,
        conversation_id: str | None = None,
        session_system_prompt: str | None = None,
    ) -> ChatSession:
        """
        conversation_id is a 256-bit string created by uuid.uuid4() and is also
        the file name under session_cache_folder/ for each conversation
        """
        return ChatSession(self, conversation_id, session_system_prompt)

    def build_messages(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        former_messages: list[dict] | None = None,
        *,
        shrink_multiple_break: bool = False,
    ) -> list[dict]:
        """build the messages to avoid implementing several redundant lines of code"""
        if former_messages is None:
            former_messages = []
        # shrink multiple break will recursively remove multiple breaks(more than 2)
        if shrink_multiple_break:
            while "\n\n\n" in user_prompt:
                user_prompt = user_prompt.replace("\n\n\n", "\n\n")
            if system_prompt is not None:
                while "\n\n\n" in system_prompt:
                    system_prompt = system_prompt.replace("\n\n\n", "\n\n")
        system_prompt = self.cfg.default_system_prompt if system_prompt is None else system_prompt
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        messages.extend(former_messages[-1 * self.cfg.max_past_message_include :])
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        return messages

    def build_messages_and_create_chat_completion(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        former_messages: list | None = None,
        chat_cache_prefix: str = "",
        *,
        shrink_multiple_break: bool = False,
        **kwargs: Any,
    ) -> str:
        if former_messages is None:
            former_messages = []
        messages = self.build_messages(user_prompt, system_prompt, former_messages, shrink_multiple_break)
        return self._try_create_chat_completion_or_embedding(
            messages=messages,
            chat_completion=True,
            chat_cache_prefix=chat_cache_prefix,
            **kwargs,
        )



    def create_embedding(self, input_content: str | list[str], **kwargs: Any) -> list[Any] | Any:
        input_content_list = [input_content] if isinstance(input_content, str) else input_content
        resp = self._try_create_chat_completion_or_embedding(
            input_content_list=input_content_list, embedding=True, **kwargs,
        )
        if isinstance(input_content, str):
            return resp[0]
        return resp

    def _create_chat_completion_auto_continue(self, messages: list, **kwargs: dict) -> str:
        """
        Call the chat completion function and automatically continue the conversation if the finish_reason is length.
        TODO: This function only continues once, maybe need to continue more than once in the future.
        """
        response, finish_reason = self._create_chat_completion_inner_function(messages=messages, **kwargs)

        if finish_reason == "length":
            new_message = deepcopy(messages)
            new_message.append({"role": "assistant", "content": response})
            new_message.append(
                {
                    "role": "user",
                    "content": "continue the former output with no overlap",
                },
            )
            new_response, finish_reason = self._create_chat_completion_inner_function(messages=new_message, **kwargs)
            return response + new_response
        return response

    def _try_create_chat_completion_or_embedding(
        self, max_retry: int = 10, *, chat_completion: bool = False, embedding: bool = False, **kwargs: Any,
    ) -> Any:
        assert not (chat_completion and embedding), "chat_completion and embedding cannot be True at the same time"
        max_retry = self.cfg.max_retry if self.cfg.max_retry is not None else max_retry
        for i in range(max_retry):
            try:
                if embedding:
                    return self._create_embedding_inner_function(**kwargs)
                if chat_completion:
                    return self._create_chat_completion_auto_continue(**kwargs)
            except openai.BadRequestError as e: # noqa: PERF203
                print(e)
                print(f"Retrying {i+1}th time...")
                if "'messages' must contain the word 'json' in some form" in e.message:
                    kwargs["add_json_in_prompt"] = True
                elif embedding and "maximum context length" in e.message:
                    kwargs["input_content_list"] = [
                        content[: len(content) // 2] for content in kwargs.get("input_content_list", [])
                    ]
            except Exception as e: # noqa: BLE001
                print(e)
                print(f"Retrying {i+1}th time...")
                time.sleep(self.retry_wait_seconds)
        error_message = f"Failed to create chat completion after {max_retry} retries."
        raise RuntimeError(error_message)

    def _create_embedding_inner_function(self, input_content_list: list[str], **kwargs: Any) -> list[Any]: # noqa: ARG002
        content_to_embedding_dict = {}
        filtered_input_content_list = []
        if self.use_embedding_cache:
            for content in input_content_list:
                cache_result = self.cache.embedding_get(content)
                if cache_result is not None:
                    content_to_embedding_dict[content] = cache_result
                else:
                    filtered_input_content_list.append(content)
        else:
            filtered_input_content_list = input_content_list

        if len(filtered_input_content_list) > 0:
            if self.use_azure:
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=filtered_input_content_list,
                )
            else:
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=filtered_input_content_list,
                )
            for index, data in enumerate(response.data):
                content_to_embedding_dict[filtered_input_content_list[index]] = data.embedding

            if self.dump_embedding_cache:
                self.cache.embedding_set(content_to_embedding_dict)
        return [content_to_embedding_dict[content] for content in input_content_list]


    def _build_messages(self, messages: list[dict]) -> str:
        log_messages = ""
        for m in messages:
            log_messages += (
                f"\n{LogColors.MAGENTA}{LogColors.BOLD}Role:{LogColors.END}"
                f"{LogColors.CYAN}{m['role']}{LogColors.END}\n"
                f"{LogColors.MAGENTA}{LogColors.BOLD}Content:{LogColors.END} "
                f"{LogColors.CYAN}{m['content']}{LogColors.END}\n"
            )
        return log_messages

    def log_messages(self, messages: list[dict]) -> None:
        if self.cfg.log_llm_chat_content:
            FinCoLog().info(self._build_messages(messages))

    def log_response(self, response: str | None = None, *, stream: bool = False) -> None:
        if self.cfg.log_llm_chat_content:
            if stream:
                FinCoLog().info(f"\n{LogColors.CYAN}Response:{LogColors.END}")
            else:
                FinCoLog().info(f"\n{LogColors.CYAN}Response:{response}{LogColors.END}")

    def _create_chat_completion_inner_function( # noqa: C901, PLR0912, PLR0915
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        chat_cache_prefix: str = "",
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        *,
        json_mode: bool = False,
        add_json_in_prompt: bool = False,
    ) -> str:
        self.log_messages(messages)
        # TODO: fail to use loguru adaptor due to stream response
        input_content_json = json.dumps(messages)
        input_content_json = (
            chat_cache_prefix + input_content_json
        )  # FIXME this is a hack to make sure the cache represents the round index
        if self.use_chat_cache:
            cache_result = self.cache.chat_get(input_content_json)
            if cache_result is not None:
                return cache_result, None

        if temperature is None:
            temperature = self.cfg.chat_temperature
        if max_tokens is None:
            max_tokens = self.cfg.chat_max_tokens
        if frequency_penalty is None:
            frequency_penalty = self.cfg.chat_frequency_penalty
        if presence_penalty is None:
            presence_penalty = self.cfg.chat_presence_penalty

        finish_reason = None
        if self.use_llama2:
            response = self.generator.chat_completion(
                messages,  # type: ignore
                max_gen_len=max_tokens,
                temperature=temperature,
            )
            resp = response[0]["generation"]["content"]
            self.log_response(resp)
        elif self.use_gcr_endpoint:
            body = str.encode(
                json.dumps(
                    {
                        "input_data": {
                            "input_string": messages,
                            "parameters": {
                                "temperature": self.gcr_endpoint_temperature,
                                "top_p": self.gcr_endpoint_top_p,
                                "do_sample": self.gcr_endpoint_do_sample,
                                "max_new_tokens": self.gcr_endpoint_max_token,
                            },
                        },
                    },
                ),
            )

            req = urllib.request.Request(self.gcr_endpoint, body, self.headers) # noqa: S310
            response = urllib.request.urlopen(req) # noqa: S310
            resp = json.loads(response.read().decode())["output"]
            self.log_response(resp)
        else:
            if self.use_azure:
                if json_mode:
                    if add_json_in_prompt:
                        for message in messages[::-1]:
                            message["content"] = message["content"] + "\nPlease respond in json format."
                            if message["role"] == "system":
                                break
                    response = self.chat_client.chat.completions.create(
                        model=self.chat_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        stream=self.chat_stream,
                        seed=self.chat_seed,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                    )
                else:
                    response = self.chat_client.chat.completions.create(
                        model=self.chat_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=self.chat_stream,
                        seed=self.chat_seed,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                    )
            else:
                response = self.chat_client.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    stream=self.chat_stream,
                    seed=self.chat_seed,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
            if self.chat_stream:
                self.log_response(stream=True)
                resp = ""
                for chunk in response:
                    content = (
                        chunk.choices[0].delta.content
                        if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None
                        else ""
                    )
                    if self.cfg.log_llm_chat_content:
                        print(LogColors.CYAN + content, end="")
                    resp += content
                    if len(chunk.choices) > 0 and chunk.choices[0].finish_reason is not None:
                        finish_reason = chunk.choices[0].finish_reason
            else:
                resp = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                self.log_response(resp)
            if json_mode:
                json.loads(resp)
        if self.dump_chat_cache:
            self.cache.chat_set(input_content_json, resp)
        # TODO: fail to use loguru adaptor due to stream response
        return resp, finish_reason

    def calculate_token_from_messages(self, messages: list[dict]) -> int:
        if self.use_llama2 or self.use_gcr_endpoint:
            FinCoLog().warning("num_tokens_from_messages() is not implemented for model llama2.")
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

    def build_messages_and_calculate_token(
        self,
        user_prompt: str,
        system_prompt: str | None,
        former_messages: list[dict] | None = None,
        *,
        shrink_multiple_break: bool = False,
    ) -> int:
        if former_messages is None:
            former_messages = []
        messages = self.build_messages(user_prompt, system_prompt, former_messages, shrink_multiple_break)
        return self.calculate_token_from_messages(messages)


def calculate_embedding_process(str_list: list) -> list:
    return APIBackend().create_embedding(str_list)


def create_embedding_with_multiprocessing(str_list: list, slice_count: int = 50, nproc: int = 8) -> list:
    embeddings = []

    pool = multiprocessing.Pool(nproc)
    result_list = [pool.apply_async(calculate_embedding_process, (str_list[index : index + slice_count],))
                    for index in range(0, len(str_list), slice_count)]
    pool.close()
    pool.join()

    for res in result_list:
        embeddings.extend(res.get())
    return embeddings



def calculate_embedding_distance_between_str_list(
    source_str_list: list[str], target_str_list: list[str],
) -> list[list[float]]:
    if not source_str_list or not target_str_list:
        return [[]]

    embeddings = create_embedding_with_multiprocessing(source_str_list + target_str_list, slice_count=50, nproc=8)
    source_embeddings = embeddings[:len(source_str_list)]
    target_embeddings = embeddings[len(source_str_list):]

    source_embeddings_np = np.array(source_embeddings)
    target_embeddings_np = np.array(target_embeddings)

    source_embeddings_np = source_embeddings_np / np.linalg.norm(source_embeddings_np, axis=1, keepdims=True)
    target_embeddings_np = target_embeddings_np / np.linalg.norm(target_embeddings_np, axis=1, keepdims=True)
    similarity_matrix = np.dot(source_embeddings_np, target_embeddings_np.T)

    return similarity_matrix.tolist()
