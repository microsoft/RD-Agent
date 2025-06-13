import copyreg
from typing import Any, Literal, cast

import numpy as np
from litellm import (
    BadRequestError,
    completion,
    completion_cost,
    embedding,
    supports_response_schema,
    token_counter,
)

from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import APIBackend
from rdagent.oai.llm_conf import LLMSettings


# NOTE: Patching! Otherwise, the exception will call the constructor and with following error:
# `BadRequestError.__init__() missing 2 required positional arguments: 'model' and 'llm_provider'`
def _reduce_no_init(exc: Exception) -> tuple:
    cls = exc.__class__
    return (cls.__new__, (cls,), exc.__dict__)


# suppose you want to apply this to MyError
copyreg.pickle(BadRequestError, _reduce_no_init)


class LiteLLMSettings(LLMSettings):
    class Config:
        env_prefix = "LITELLM_"

    """Use `LITELLM_` as prefix for environment variables"""

    # Placeholder for LiteLLM specific settings, so far it's empty


LITELLM_SETTINGS = LiteLLMSettings()
logger.info(f"{LITELLM_SETTINGS}")
ACC_COST = 0.0


class LiteLLMAPIBackend(APIBackend):
    """LiteLLM implementation of APIBackend interface"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_chat_api_config(self) -> dict[str, Any]:
        """获取Chat API的配置参数"""
        config = {}

        # 优先级：chat专用配置 > 通用配置 > 环境变量
        api_key = LITELLM_SETTINGS.chat_openai_api_key or LITELLM_SETTINGS.openai_api_key
        base_url = LITELLM_SETTINGS.chat_openai_base_url

        if api_key:
            config["api_key"] = api_key
        if base_url:
            config["api_base"] = base_url

        return config

    def _get_embedding_api_config(self) -> dict[str, Any]:
        """获取Embedding API的配置参数"""
        config = {}

        # 优先级：embedding专用配置 > 通用配置 > 环境变量
        api_key = LITELLM_SETTINGS.embedding_openai_api_key or LITELLM_SETTINGS.openai_api_key
        base_url = LITELLM_SETTINGS.embedding_openai_base_url

        if api_key:
            config["api_key"] = api_key
        if base_url:
            config["api_base"] = base_url

        return config

    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        """
        Calculate the token count from messages
        """
        num_tokens = token_counter(
            model=LITELLM_SETTINGS.chat_model,
            messages=messages,
        )
        logger.info(f"{LogColors.CYAN}Token count: {LogColors.END} {num_tokens}", tag="debug_litellm_token")
        return num_tokens

    def _create_embedding_inner_function(
        self, input_content_list: list[str], *args: Any, **kwargs: Any
    ) -> list[list[float]]:
        model_name = LITELLM_SETTINGS.embedding_model

        # 获取embedding专用的API配置
        embedding_config = self._get_embedding_api_config()

        # 合并配置参数
        call_kwargs = {
            "model": model_name,
            "input": input_content_list,
            **embedding_config,  # 添加API密钥和base URL
            **kwargs,  # 用户传入的其他参数
        }

        logger.info(f"{LogColors.GREEN}Using embedding model{LogColors.END} {model_name}", tag="llm_messages")
        if embedding_config.get("api_base"):
            logger.info(
                f"{LogColors.GREEN}Using embedding base URL{LogColors.END} {embedding_config['api_base']}",
                tag="llm_messages",
            )

        response = embedding(**call_kwargs)
        response_list = [data["embedding"] for data in response.data]
        return response_list

    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def] # noqa: C901, PLR0912, PLR0915
        self,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
        *args,
        **kwargs,
    ) -> tuple[str, str | None]:
        """
        Call the chat completion function
        """
        if json_mode and supports_response_schema(model=LITELLM_SETTINGS.chat_model):
            kwargs["response_format"] = {"type": "json_object"}

        logger.info(self._build_log_messages(messages), tag="llm_messages")

        # Call LiteLLM completion
        model = LITELLM_SETTINGS.chat_model
        temperature = LITELLM_SETTINGS.chat_temperature
        max_tokens = LITELLM_SETTINGS.chat_max_tokens
        reasoning_effort = LITELLM_SETTINGS.reasoning_effort

        if LITELLM_SETTINGS.chat_model_map:
            for t, mc in LITELLM_SETTINGS.chat_model_map.items():
                if t in logger._tag:
                    model = mc["model"]
                    if "temperature" in mc:
                        temperature = float(mc["temperature"])
                    if "max_tokens" in mc:
                        max_tokens = int(mc["max_tokens"])
                    if "reasoning_effort" in mc:
                        if mc["reasoning_effort"] in ["low", "medium", "high"]:
                            reasoning_effort = cast(Literal["low", "medium", "high"], mc["reasoning_effort"])
                        else:
                            reasoning_effort = None
                    break

        # 获取chat专用的API配置
        chat_config = self._get_chat_api_config()

        # 合并配置参数
        call_kwargs = {
            "model": model,
            "messages": messages,
            "stream": LITELLM_SETTINGS.chat_stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "max_retries": 0,
            **chat_config,  # 添加API密钥和base URL
            **kwargs,  # 用户传入的其他参数
        }

        response = completion(**call_kwargs)
        logger.info(f"{LogColors.GREEN}Using chat model{LogColors.END} {model}", tag="llm_messages")
        if chat_config.get("api_base"):
            logger.info(
                f"{LogColors.GREEN}Using chat base URL{LogColors.END} {chat_config['api_base']}", tag="llm_messages"
            )

        if LITELLM_SETTINGS.chat_stream:
            logger.info(f"{LogColors.BLUE}assistant:{LogColors.END}", tag="llm_messages")
            content = ""
            finish_reason = None
            for message in response:
                if message["choices"][0]["finish_reason"]:
                    finish_reason = message["choices"][0]["finish_reason"]
                if "content" in message["choices"][0]["delta"]:
                    chunk = (
                        message["choices"][0]["delta"]["content"] or ""
                    )  # when finish_reason is "stop", content is None
                    content += chunk
                    logger.info(LogColors.CYAN + chunk + LogColors.END, raw=True, tag="llm_messages")

            logger.info("\n", raw=True, tag="llm_messages")
        else:
            content = str(response.choices[0].message.content)
            finish_reason = response.choices[0].finish_reason
            finish_reason_str = (
                f"({LogColors.RED}Finish reason: {finish_reason}{LogColors.END})"
                if finish_reason and finish_reason != "stop"
                else ""
            )
            logger.info(f"{LogColors.BLUE}assistant:{LogColors.END} {finish_reason_str}\n{content}", tag="llm_messages")

        global ACC_COST
        try:
            cost = completion_cost(model=model, messages=messages, completion=content)
        except Exception as e:
            logger.warning(f"Cost calculation failed for model {model}: {e}. Skip cost statistics.")
            cost = np.nan
        else:
            ACC_COST += cost
            logger.info(
                f"Current Cost: ${float(cost):.10f}; Accumulated Cost: ${float(ACC_COST):.10f}; {finish_reason=}",
            )

        prompt_tokens = token_counter(model=model, messages=messages)
        completion_tokens = token_counter(model=model, text=content)
        logger.log_object(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "accumulated_cost": ACC_COST,
            },
            tag="token_cost",
        )
        return content, finish_reason
