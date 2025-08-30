import copyreg
from typing import Any, Literal, Optional, Type, Union, cast

import numpy as np
from litellm import (
    BadRequestError,
    completion,
    completion_cost,
    embedding,
    get_max_tokens,
    supports_function_calling,
    supports_response_schema,
    token_counter,
)
from pydantic import BaseModel

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
ACC_COST = 0.0


class LiteLLMAPIBackend(APIBackend):
    """LiteLLM implementation of APIBackend interface"""

    _has_logged_settings: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not self.__class__._has_logged_settings:
            logger.info(f"{LITELLM_SETTINGS}")
            logger.log_object(LITELLM_SETTINGS.model_dump(), tag="LITELLM_SETTINGS")
            self.__class__._has_logged_settings = True
        super().__init__(*args, **kwargs)

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

    def _create_embedding_inner_function(self, input_content_list: list[str]) -> list[list[float]]:
        """
        Call the embedding function
        """
        model_name = LITELLM_SETTINGS.embedding_model
        logger.info(f"{LogColors.GREEN}Using emb model{LogColors.END} {model_name}", tag="debug_litellm_emb")
        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(
                f"{LogColors.MAGENTA}Creating embedding{LogColors.END} for: {input_content_list}",
                tag="debug_litellm_emb",
            )
        response = embedding(
            model=model_name,
            input=input_content_list,
        )
        response_list = [data["embedding"] for data in response.data]
        return response_list

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

        if response_format and not supports_response_schema(model=LITELLM_SETTINGS.chat_model):
            # Deepseek will enter this branch
            logger.warning(
                f"{LogColors.YELLOW}Model {LITELLM_SETTINGS.chat_model} does not support response schema, ignoring response_format argument.{LogColors.END}",
                tag="llm_messages",
            )
            response_format = None

        if response_format:
            kwargs["response_format"] = response_format

        if LITELLM_SETTINGS.log_llm_chat_content:
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
        response = completion(
            model=model,
            messages=messages,
            stream=LITELLM_SETTINGS.chat_stream,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            max_retries=0,
            **kwargs,
        )
        logger.info(f"{LogColors.GREEN}Using chat model{LogColors.END} {model}", tag="llm_messages")

        if LITELLM_SETTINGS.chat_stream:
            if LITELLM_SETTINGS.log_llm_chat_content:
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
                    if LITELLM_SETTINGS.log_llm_chat_content:
                        logger.info(LogColors.CYAN + chunk + LogColors.END, raw=True, tag="llm_messages")
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info("\n", raw=True, tag="llm_messages")
        else:
            content = str(response.choices[0].message.content)
            finish_reason = response.choices[0].finish_reason
            finish_reason_str = (
                f"({LogColors.RED}Finish reason: {finish_reason}{LogColors.END})"
                if finish_reason and finish_reason != "stop"
                else ""
            )
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(
                    f"{LogColors.BLUE}assistant:{LogColors.END} {finish_reason_str}\n{content}", tag="llm_messages"
                )

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

    def supports_response_schema(self) -> bool:
        """
        Check if the backend supports function calling
        """
        return supports_response_schema(model=LITELLM_SETTINGS.chat_model) and LITELLM_SETTINGS.enable_response_schema

    @property
    def chat_token_limit(self) -> int:
        try:
            max_tokens = get_max_tokens(LITELLM_SETTINGS.chat_model)
            if max_tokens is None:
                return super().chat_token_limit
            return max_tokens
        except Exception as e:
            return super().chat_token_limit

    def supports_function_calling(self) -> bool:
        """
        Check if the backend supports function calling
        """
        return supports_function_calling(model=LITELLM_SETTINGS.chat_model)

    def convert_mcp_tools_to_openai_format(self, mcp_tools: list) -> list[dict[str, Any]]:
        """
        Convert MCP tools to OpenAI function calling format

        Args:
            mcp_tools: List of MCP Tool objects

        Returns:
            List of tools in OpenAI format
        """
        openai_tools = []
        for tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            openai_tools.append(openai_tool)

        # Conversion completed silently
        return openai_tools

    def call_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any
    ) -> tuple[str, str | None, list[Any] | None]:
        """
        Call chat completion with tools support

        Args:
            messages: Conversation messages
            tools: Tools in OpenAI format
            **kwargs: Additional parameters

        Returns:
            Tuple of (content, finish_reason, tool_calls)
        """
        # Check if we actually need function calling
        if tools:
            # Tools provided, check if model supports function calling
            if not self.supports_function_calling():
                logger.warning(
                    f"Model {LITELLM_SETTINGS.chat_model} does not support function calling", tag="litellm_tools"
                )
                # Fall back to regular chat completion
                content, finish_reason = self._create_chat_completion_inner_function(messages, **kwargs)
                return content, finish_reason, None

            # Model supports function calling and tools are provided
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Model call with tools initiated

        # Build parameters with potential overrides from kwargs
        params = {
            "model": kwargs.pop("model", LITELLM_SETTINGS.chat_model),
            "messages": messages,
            "stream": False,  # Tools don't work well with streaming
            "temperature": kwargs.pop("temperature", LITELLM_SETTINGS.chat_temperature),
            "max_tokens": kwargs.pop("max_tokens", LITELLM_SETTINGS.chat_max_tokens),
            "max_retries": 0,
            **kwargs,  # Include remaining kwargs
        }

        # Use existing chat completion infrastructure
        response = completion(**params)

        assistant_message = response.choices[0].message
        content = assistant_message.content or ""
        finish_reason = response.choices[0].finish_reason
        tool_calls = getattr(assistant_message, "tool_calls", None)

        # Log and calculate cost using existing infrastructure
        global ACC_COST
        try:
            cost = completion_cost(model=LITELLM_SETTINGS.chat_model, messages=messages, completion=content)
            ACC_COST += cost
            logger.info(
                f"Tool-enabled call - Current Cost: ${float(cost):.10f}; "
                f"Accumulated Cost: ${float(ACC_COST):.10f}; {finish_reason=}",
                tag="litellm_tools",
            )
        except Exception as e:
            logger.warning(f"Cost calculation failed for tools call: {e}")

        # Tool calls processed silently

        return content, finish_reason, tool_calls

    async def multi_round_tool_calling(
        self,
        initial_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_rounds: int = 5,
        tool_executor: Any = None,
        verbose: bool = False,
        model_config_override: dict[str, Any] | None = None,
        round_callback: Any = None,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Perform multi-round tool calling conversation

        Args:
            initial_messages: Initial conversation messages
            tools: Available tools in OpenAI format
            max_rounds: Maximum number of rounds
            tool_executor: Function to execute tool calls
            verbose: Enable verbose logging
            model_config_override: Override model configuration (model, api_base, api_key, etc.)
            round_callback: Optional async callback(round_num, messages) called after each round
            **kwargs: Additional parameters for chat completion

        Returns:
            Tuple of (final_response, full_conversation)
        """
        if not self.supports_function_calling():
            logger.warning("Multi-round tool calling not supported by this model", tag="litellm_tools")
            # Fall back to single completion
            content, _ = self._create_chat_completion_inner_function(initial_messages, **kwargs)
            return content, initial_messages

        messages = initial_messages.copy()

        # Apply model configuration override if provided
        if model_config_override:
            for key, value in model_config_override.items():
                if key == "api_base":
                    kwargs["base_url"] = value
                elif key in ["model", "api_key", "temperature", "max_tokens"]:
                    kwargs[key] = value

        last_finish_reason = None

        for round_count in range(1, max_rounds + 1):
            if verbose:
                logger.info(f"🔄 Round {round_count}/{max_rounds}", tag="mcp_progress")

            # Call with tools (now with potential overrides applied)
            content, finish_reason, tool_calls = self.call_with_tools(messages, tools, **kwargs)
            last_finish_reason = finish_reason  # Track finish_reason

            # Add assistant response to conversation
            assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_message)

            # If no tool calls, we're done
            if not tool_calls:
                if verbose:
                    logger.info(f"✅ Final response in round {round_count}", tag="mcp_progress")
                # Call round callback before returning
                if round_callback:
                    try:
                        await round_callback(round_count, messages.copy())
                    except Exception as e:
                        logger.warning(f"Round callback error: {e}", tag="litellm_callback")
                return content, messages

            # Execute tool calls if executor provided
            if tool_executor:
                if verbose:
                    tool_names = [tc.function.name for tc in tool_calls]
                    tool_list = ", ".join(tool_names)
                    logger.info(f"🔧 Executing {len(tool_calls)} tool(s): {tool_list}", tag="mcp_progress")

                tool_results = await tool_executor(tool_calls)
                messages.extend(tool_results)
            else:
                # If no executor, add placeholder results
                for tool_call in tool_calls:
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": "Tool execution not configured"}
                    )

            # Call round callback if provided
            if round_callback:
                try:
                    await round_callback(round_count, messages.copy())
                except Exception as e:
                    logger.warning(f"Round callback error: {e}", tag="litellm_callback")

            # Round completion logged at higher level

        # Reached max rounds - check if we need a final answer
        if last_finish_reason == "tool_calls":
            logger.warning(
                f"Reached max rounds with tool_calls, making final call without tools for answer", tag="mcp_progress"
            )

            # Add a system message to prompt for final answer
            messages.append(
                {
                    "role": "system",
                    "content": "Based on all the tool results above, please provide a comprehensive final answer to the user's original question.",
                }
            )

            # Make final call without tools to get text answer
            # Use call_with_tools with empty tools list to force text response
            content, finish_reason, _ = self.call_with_tools(
                messages, tools=[], **kwargs  # Empty tools list forces pure text response
            )
            final_content = content

            messages.append({"role": "assistant", "content": final_content})

            return final_content, messages
        else:
            # finish_reason was 'stop' or other, return last response
            logger.info(f"Reached max rounds with finish_reason='{last_finish_reason}'", tag="mcp_progress")
            return messages[-1].get("content", "No response generated"), messages
