from __future__ import annotations

import io
import json
import re
import sqlite3
import time
import tokenize
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Type, Union, cast

import pytz
from pydantic import BaseModel, TypeAdapter

from rdagent.core.exception import PolicyError
from rdagent.core.utils import LLM_CACHE_SEED_GEN, SingletonBaseClass
from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.log.timer import RD_Agent_TIMER_wrapper
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.utils import md5_hash

try:
    import litellm
    import openai

    openai_imported = True
except ImportError:
    openai_imported = False


class JSONParser:
    """JSON parser supporting multiple strategies"""

    def __init__(self, add_json_in_prompt: bool = False) -> None:
        self.strategies: List[Callable[[str], str]] = [
            self._direct_parse,
            self._extract_from_code_block,
            self._fix_python_syntax,
            self._extract_with_fix_combined,
        ]
        self.add_json_in_prompt = add_json_in_prompt

    def parse(self, content: str) -> str:
        """Parse JSON content, automatically trying multiple strategies"""
        original_content = content

        for strategy in self.strategies:
            try:
                return strategy(original_content)
            except json.JSONDecodeError:
                continue

        # All strategies failed
        if not self.add_json_in_prompt:
            error = json.JSONDecodeError(
                "Failed to parse JSON after all attempts, maybe because 'messages' must contain the word 'json' in some form",
                original_content,
                0,
            )
            error.message = "Failed to parse JSON after all attempts, maybe because 'messages' must contain the word 'json' in some form"  # type: ignore[attr-defined]
            raise error
        else:
            raise json.JSONDecodeError("Failed to parse JSON after all attempts", original_content, 0)

    def _direct_parse(self, content: str) -> str:
        """Strategy 1: Direct parsing (including handling extra data)"""
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                return self._extract_first_json(content)
            raise

    def _extract_from_code_block(self, content: str) -> str:
        """Strategy 2: Extract JSON from code block"""
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON code block found", content, 0)

        json_content = match.group(1).strip()
        return self._direct_parse(json_content)

    def _fix_python_syntax(self, content: str) -> str:
        """Strategy 3: Fix Python syntax before parsing"""
        fixed = self._fix_python_booleans(content)
        return self._direct_parse(fixed)

    def _extract_with_fix_combined(self, content: str) -> str:
        """Strategy 4: Combined strategy - fix Python syntax first, then extract the first JSON object"""
        fixed = self._fix_python_booleans(content)

        # Try to extract code block from the fixed content
        match = re.search(r"```json\s*(.*?)\s*```", fixed, re.DOTALL)
        if match:
            fixed = match.group(1).strip()

        return self._direct_parse(fixed)

    @staticmethod
    def _fix_python_booleans(json_str: str) -> str:
        """Safely fix Python-style booleans to JSON standard format using tokenize"""
        replacements = {"True": "true", "False": "false", "None": "null"}

        try:
            out = []
            io_string = io.StringIO(json_str)
            tokens = tokenize.generate_tokens(io_string.readline)

            for toknum, tokval, _, _, _ in tokens:
                if toknum == tokenize.NAME and tokval in replacements:
                    out.append(replacements[tokval])
                else:
                    out.append(tokval)

            result = "".join(out)
            return result

        except (tokenize.TokenError, json.JSONDecodeError):
            # If tokenize fails, fallback to regex method
            for python_val, json_val in replacements.items():
                json_str = re.sub(rf"\b{python_val}\b", json_val, json_str)
            return json_str

    @staticmethod
    def _extract_first_json(response: str) -> str:
        """Extract the first complete JSON object, ignoring extra content"""
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(response)
        return json.dumps(obj)


class SQliteLazyCache(SingletonBaseClass):
    def __init__(self, cache_location: str) -> None:
        super().__init__()
        self.cache_location = cache_location
        db_file_exist = Path(cache_location).exists()
        # TODO: sqlite3 does not support multiprocessing.
        self.conn = sqlite3.connect(cache_location, timeout=20)
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
            self.c.execute(
                """
                CREATE TABLE message_cache (
                    conversation_id TEXT PRIMARY KEY,
                    message TEXT
                )
                """,
            )
            self.conn.commit()

    def chat_get(self, key: str) -> str | None:
        md5_key = md5_hash(key)
        self.c.execute("SELECT chat FROM chat_cache WHERE md5_key=?", (md5_key,))
        result = self.c.fetchone()
        return None if result is None else result[0]

    def embedding_get(self, key: str) -> list | dict | str | None:
        md5_key = md5_hash(key)
        self.c.execute("SELECT embedding FROM embedding_cache WHERE md5_key=?", (md5_key,))
        result = self.c.fetchone()
        return None if result is None else json.loads(result[0])

    def chat_set(self, key: str, value: str) -> None:
        md5_key = md5_hash(key)
        self.c.execute(
            "INSERT OR REPLACE INTO chat_cache (md5_key, chat) VALUES (?, ?)",
            (md5_key, value),
        )
        self.conn.commit()
        return None

    def embedding_set(self, content_to_embedding_dict: dict) -> None:
        for key, value in content_to_embedding_dict.items():
            md5_key = md5_hash(key)
            self.c.execute(
                "INSERT OR REPLACE INTO embedding_cache (md5_key, embedding) VALUES (?, ?)",
                (md5_key, json.dumps(value)),
            )
        self.conn.commit()

    def message_get(self, conversation_id: str) -> list[dict[str, Any]]:
        self.c.execute("SELECT message FROM message_cache WHERE conversation_id=?", (conversation_id,))
        result = self.c.fetchone()
        return [] if result is None else cast(list[dict[str, Any]], json.loads(result[0]))

    def message_set(self, conversation_id: str, message_value: list[dict[str, Any]]) -> None:
        self.c.execute(
            "INSERT OR REPLACE INTO message_cache (conversation_id, message) VALUES (?, ?)",
            (conversation_id, json.dumps(message_value)),
        )
        self.conn.commit()
        return None


class SessionChatHistoryCache(SingletonBaseClass):
    def __init__(self) -> None:
        """load all history conversation json file from self.session_cache_location"""
        self.cache = SQliteLazyCache(cache_location=LLM_SETTINGS.prompt_cache_path)

    def message_get(self, conversation_id: str) -> list[dict[str, Any]]:
        return self.cache.message_get(conversation_id)

    def message_set(self, conversation_id: str, message_value: list[dict[str, Any]]) -> None:
        self.cache.message_set(conversation_id, message_value)


class ChatSession:
    def __init__(self, api_backend: Any, conversation_id: str | None = None, system_prompt: str | None = None) -> None:
        self.conversation_id = str(uuid.uuid4()) if conversation_id is None else conversation_id
        self.system_prompt = system_prompt if system_prompt is not None else LLM_SETTINGS.default_system_prompt
        self.api_backend = api_backend

    def build_chat_completion_message(self, user_prompt: str) -> list[dict[str, Any]]:
        history_message = SessionChatHistoryCache().message_get(self.conversation_id)
        messages = history_message
        if not messages:
            messages.append({"role": LLM_SETTINGS.system_prompt_role, "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        return messages

    def build_chat_completion_message_and_calculate_token(self, user_prompt: str) -> Any:
        messages = self.build_chat_completion_message(user_prompt)
        return self.api_backend._calculate_token_from_messages(messages)

    def build_chat_completion(self, user_prompt: str, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        """
        this function is to build the session messages
        user prompt should always be provided
        """
        messages = self.build_chat_completion_message(user_prompt)

        with logger.tag(f"session_{self.conversation_id}"):
            response: str = self.api_backend._try_create_chat_completion_or_embedding(  # noqa: SLF001
                *args,
                messages=messages,
                chat_completion=True,
                **kwargs,
            )
            logger.log_object({"user": user_prompt, "resp": response}, tag="debug_llm")

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


class APIBackend(ABC):
    """
    Abstract base class for LLM API backends
    supporting auto retry, cache and auto continue
    Inner api call should be implemented in the subclass
    """

    def __init__(
        self,
        use_chat_cache: bool | None = None,
        dump_chat_cache: bool | None = None,
        use_embedding_cache: bool | None = None,
        dump_embedding_cache: bool | None = None,
    ):
        self.dump_chat_cache = LLM_SETTINGS.dump_chat_cache if dump_chat_cache is None else dump_chat_cache
        self.use_chat_cache = LLM_SETTINGS.use_chat_cache if use_chat_cache is None else use_chat_cache
        self.dump_embedding_cache = (
            LLM_SETTINGS.dump_embedding_cache if dump_embedding_cache is None else dump_embedding_cache
        )
        self.use_embedding_cache = (
            LLM_SETTINGS.use_embedding_cache if use_embedding_cache is None else use_embedding_cache
        )
        if self.dump_chat_cache or self.use_chat_cache or self.dump_embedding_cache or self.use_embedding_cache:
            self.cache_file_location = LLM_SETTINGS.prompt_cache_path
            self.cache = SQliteLazyCache(cache_location=self.cache_file_location)

        self.retry_wait_seconds = LLM_SETTINGS.retry_wait_seconds

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

    def _build_messages(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        former_messages: list[dict[str, Any]] | None = None,
        *,
        shrink_multiple_break: bool = False,
    ) -> list[dict[str, Any]]:
        """
        build the messages to avoid implementing several redundant lines of code

        """
        if former_messages is None:
            former_messages = []
        # shrink multiple break will recursively remove multiple breaks(more than 2)
        if shrink_multiple_break:
            while "\n\n\n" in user_prompt:
                user_prompt = user_prompt.replace("\n\n\n", "\n\n")
            if system_prompt is not None:
                while "\n\n\n" in system_prompt:
                    system_prompt = system_prompt.replace("\n\n\n", "\n\n")
        system_prompt = LLM_SETTINGS.default_system_prompt if system_prompt is None else system_prompt
        messages = [
            {
                "role": LLM_SETTINGS.system_prompt_role,
                "content": system_prompt,
            },
        ]
        messages.extend(former_messages[-1 * LLM_SETTINGS.max_past_message_include :])
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        return messages

    def _build_log_messages(self, messages: list[dict[str, Any]]) -> str:
        log_messages = ""
        for m in messages:
            log_messages += (
                f"\n{LogColors.MAGENTA}{LogColors.BOLD}Role:{LogColors.END}"
                f"{LogColors.CYAN}{m['role']}{LogColors.END}\n"
                f"{LogColors.MAGENTA}{LogColors.BOLD}Content:{LogColors.END} "
                f"{LogColors.CYAN}{m['content']}{LogColors.END}\n"
            )
        return log_messages

    def build_messages_and_create_chat_completion(  # type: ignore[no-untyped-def]
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        former_messages: list | None = None,
        chat_cache_prefix: str = "",
        shrink_multiple_break: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """
        Responseible for building messages and logging messages

        TODO: What is weird is that the function is called before we seperate embeddings and chat completion.

        Parameters
        ----------
        user_prompt : str
        system_prompt : str | None
        former_messages : list | None
        response_format : BaseModel | dict
            A BaseModel based on pydantic or a dict
        **kwargs
        Returns
        -------
        str
        """
        if former_messages is None:
            former_messages = []
        messages = self._build_messages(
            user_prompt,
            system_prompt,
            former_messages,
            shrink_multiple_break=shrink_multiple_break,
        )

        resp = self._try_create_chat_completion_or_embedding(  # type: ignore[misc]
            *args,
            messages=messages,
            chat_completion=True,
            chat_cache_prefix=chat_cache_prefix,
            **kwargs,
        )
        if isinstance(resp, list):
            raise ValueError("The response of _try_create_chat_completion_or_embedding should be a string.")
        logger.log_object({"system": system_prompt, "user": user_prompt, "resp": resp}, tag="debug_llm")
        return resp

    def create_embedding(self, input_content: str | list[str], *args, **kwargs) -> list[float] | list[list[float]]:  # type: ignore[no-untyped-def]
        input_content_list = [input_content] if isinstance(input_content, str) else input_content
        resp = self._try_create_chat_completion_or_embedding(  # type: ignore[misc]
            input_content_list=input_content_list,
            embedding=True,
            *args,
            **kwargs,
        )
        if isinstance(input_content, str):
            return resp[0]  # type: ignore[return-value]
        return resp  # type: ignore[return-value]

    def build_messages_and_calculate_token(
        self,
        user_prompt: str,
        system_prompt: str | None,
        former_messages: list[dict[str, Any]] | None = None,
        *,
        shrink_multiple_break: bool = False,
    ) -> int:
        if former_messages is None:
            former_messages = []
        messages = self._build_messages(
            user_prompt, system_prompt, former_messages, shrink_multiple_break=shrink_multiple_break
        )
        return self._calculate_token_from_messages(messages)

    def _try_create_chat_completion_or_embedding(  # type: ignore[no-untyped-def]
        self,
        max_retry: int = 10,
        chat_completion: bool = False,
        embedding: bool = False,
        *args,
        **kwargs,
    ) -> str | list[list[float]]:
        """This function to share operation between embedding and chat completion"""
        assert not (chat_completion and embedding), "chat_completion and embedding cannot be True at the same time"
        max_retry = LLM_SETTINGS.max_retry if LLM_SETTINGS.max_retry is not None else max_retry
        timeout_count = 0
        violation_count = 0
        for i in range(max_retry):
            API_start_time = datetime.now()
            try:
                if embedding:
                    return self._create_embedding_with_cache(*args, **kwargs)
                if chat_completion:
                    return self._create_chat_completion_auto_continue(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                if hasattr(e, "message") and (
                    "'messages' must contain the word 'json' in some form" in e.message
                    or "\\'messages\\' must contain the word \\'json\\' in some form" in e.message
                ):
                    kwargs["add_json_in_prompt"] = True
                elif hasattr(e, "message") and embedding and "maximum context length" in e.message:
                    kwargs["input_content_list"] = [
                        content[: len(content) // 2] for content in kwargs.get("input_content_list", [])
                    ]
                else:
                    RD_Agent_TIMER_wrapper.api_fail_count += 1
                    RD_Agent_TIMER_wrapper.latest_api_fail_time = datetime.now(pytz.timezone("Asia/Shanghai"))

                    if (
                        openai_imported
                        and isinstance(e, litellm.BadRequestError)
                        and (
                            isinstance(e.__cause__, litellm.ContentPolicyViolationError)
                            or "The response was filtered due to the prompt triggering Azure OpenAI's content management policy"
                            in str(e)
                        )
                    ):
                        violation_count += 1
                        if violation_count >= LLM_SETTINGS.violation_fail_limit:
                            logger.warning("Content policy violation detected.")
                            raise PolicyError(e)

                    if (
                        openai_imported
                        and isinstance(e, openai.APITimeoutError)
                        or (
                            isinstance(e, openai.APIError)
                            and hasattr(e, "message")
                            and "Your resource has been temporarily blocked because we detected behavior that may violate our content policy."
                            in e.message
                        )
                    ):
                        timeout_count += 1
                        if timeout_count >= LLM_SETTINGS.timeout_fail_limit:
                            logger.warning("Timeout error, please check your network connection.")
                            raise e

                    recommended_wait_seconds = self.retry_wait_seconds
                    if openai_imported and isinstance(e, openai.RateLimitError) and hasattr(e, "message"):
                        match = re.search(r"Please retry after (\d+) seconds\.", e.message)
                        if match:
                            recommended_wait_seconds = int(match.group(1))
                    time.sleep(recommended_wait_seconds)
                    if RD_Agent_TIMER_wrapper.timer.started and not isinstance(e, json.decoder.JSONDecodeError):
                        RD_Agent_TIMER_wrapper.timer.add_duration(datetime.now() - API_start_time)
                logger.warning(str(e))
                logger.warning(f"Retrying {i+1}th time...")
        error_message = f"Failed to create chat completion after {max_retry} retries."
        raise RuntimeError(error_message)

    def _add_json_in_prompt(self, messages: list[dict[str, Any]]) -> None:
        """
        add json related content in the prompt if add_json_in_prompt is True
        """
        for message in messages[::-1]:
            message["content"] = message["content"] + "\nPlease respond in json format."
            if message["role"] == LLM_SETTINGS.system_prompt_role:
                # NOTE: assumption: systemprompt is always the first message
                break

    def _create_chat_completion_auto_continue(
        self,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
        chat_cache_prefix: str = "",
        seed: Optional[int] = None,
        json_target_type: Optional[str] = None,
        add_json_in_prompt: bool = False,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the chat completion function and automatically continue the conversation if the finish_reason is length.
        """

        if response_format is None and json_mode:
            response_format = {"type": "json_object"}

        # 0) return directly if cache is hit
        if seed is None and LLM_SETTINGS.use_auto_chat_cache_seed_gen:
            seed = LLM_CACHE_SEED_GEN.get_next_seed()
        input_content_json = json.dumps(messages)
        input_content_json = (
            chat_cache_prefix + input_content_json + f"<seed={seed}/>"
        )  # FIXME this is a hack to make sure the cache represents the round index
        if self.use_chat_cache:
            cache_result = self.cache.chat_get(input_content_json)
            if cache_result is not None:
                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info(self._build_log_messages(messages), tag="llm_messages")
                    logger.info(f"{LogColors.CYAN}Response:{cache_result}{LogColors.END}", tag="llm_messages")
                return cache_result

        # 1) get a full response
        all_response = ""
        new_messages = deepcopy(messages)
        # Loop to get a full response
        try_n = 6
        for _ in range(try_n):  # for some long code, 3 times may not enough for reasoning models
            if response_format == {"type": "json_object"} and add_json_in_prompt:
                self._add_json_in_prompt(new_messages)
            response, finish_reason = self._create_chat_completion_inner_function(
                messages=new_messages,
                response_format=response_format,
                **kwargs,
            )
            all_response += response
            if finish_reason is None or finish_reason != "length":
                break  # we get a full response now.
            new_messages.append({"role": "assistant", "content": response})
        else:
            raise RuntimeError(f"Failed to continue the conversation after {try_n} retries.")

        # 2) refine the response and return
        if LLM_SETTINGS.reasoning_think_rm:
            # Strategy 1: Try to match complete <think>...</think> pattern
            match = re.search(r"<think>(.*?)</think>(.*)", all_response, re.DOTALL)
            if match:
                _, all_response = match.groups()
            else:
                # Strategy 2: If no complete match, try to match only </think>
                match = re.search(r"</think>(.*)", all_response, re.DOTALL)
                if match:
                    all_response = match.group(1)
                # If no match at all, keep original content

        # 3) format checking
        if response_format == {"type": "json_object"} or json_target_type:
            parser = JSONParser(add_json_in_prompt=add_json_in_prompt)
            all_response = parser.parse(all_response)
            if json_target_type:
                # deepseek will enter this branch
                TypeAdapter(json_target_type).validate_json(all_response)

        if response_format is not None:
            if not isinstance(response_format, dict) and issubclass(response_format, BaseModel):
                # It may raise TypeError if initialization fails
                response_format(**json.loads(all_response))
            elif response_format == {"type": "json_object"}:
                logger.info(f"Using OpenAI response format: {response_format}")
            else:
                logger.warning(f"Unknown response_format: {response_format}, skipping validation.")
        if self.dump_chat_cache:
            self.cache.chat_set(input_content_json, all_response)
        return all_response

    def _create_embedding_with_cache(
        self, input_content_list: list[str], *args: Any, **kwargs: Any
    ) -> list[list[float]]:
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
            resp = self._create_embedding_inner_function(input_content_list=filtered_input_content_list)
            for index, data in enumerate(resp):
                content_to_embedding_dict[filtered_input_content_list[index]] = data
            if self.dump_embedding_cache:
                self.cache.embedding_set(content_to_embedding_dict)
        return [content_to_embedding_dict[content] for content in input_content_list]  # type: ignore[misc]

    @abstractmethod
    def supports_response_schema(self) -> bool:
        """
        Check if the backend supports function calling
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        """
        Calculate the token count from messages
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _create_embedding_inner_function(  # type: ignore[no-untyped-def]
        self, input_content_list: list[str], *args, **kwargs
    ) -> list[list[float]]:  # noqa: ARG002
        """
        Call the embedding function
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def] # noqa: C901, PLR0912, PLR0915
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        *args,
        **kwargs,
    ) -> tuple[str, str | None]:
        """
        Call the chat completion function
        """
        raise NotImplementedError("Subclasses must implement this method")
