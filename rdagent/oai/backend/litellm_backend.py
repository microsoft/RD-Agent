from typing import Any, Optional, Union, List, Dict
import uuid
from rdagent.oai.backend.base import APIBackend
from litellm import completion, acompletion 
from litellm import encode as encode_litellm
import os

from pathlib import Path

from rdagent.core.utils import LLM_CACHE_SEED_GEN, SingletonBaseClass, import_class
from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS

DEFAULT_QLIB_DOT_PATH = Path("./")





class LiteLLMAPIBackend(APIBackend):
    """LiteLLM implementation of APIBackend interface"""
    
    def __init__(self, litellm_model_name : str = "",litellm_api_key: str = "",*args, **kwargs) :
        super().__init__()
        def _get_encoder(text):
            return encode_litellm(model=LLM_SETTINGS.litellm_embedding_model_name or "ollama/nomic-embed-text",text=text)
        class _Encoder:
            def encode(self,text):
                return _get_encoder(text)
        self.encoder = _Encoder()
        # Set up any required LiteLLM configurations
        # if *args or **kwargs:
        if len (args) > 0 or len(kwargs) > 0:
            logger.warning("LiteLLM backend does not support any additional arguments")
    def build_chat_session(self, conversation_id: Optional[str] = None,
                         session_system_prompt: Optional[str] = None) -> Any:
        """Create a new chat session using LiteLLM"""
        # return {
        #     "conversation_id": conversation_id or str(uuid.uuid4()),
        #     "system_prompt": session_system_prompt,
        #     "messages": []
        # }
        raise NotImplementedError("LiteLLM backend does not support chat session creation")
        # TODO: Implement the chat session creation logic , with ChatSession class
        
    def build_messages_and_create_chat_completion(self, user_prompt: str,
                                                 system_prompt: Optional[str] = None,
                                                 former_messages: Optional[List] = None,
                                                 chat_cache_prefix: str = "",
                                                 shrink_multiple_break: bool = False,
                                                 *args, **kwargs) -> str:
        """Build messages and get LiteLLM chat completion"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        if former_messages:
            messages.extend(former_messages)
            
        messages.append({"role": "user", "content": user_prompt})
        
        # Call LiteLLM completion
        response = completion(
            model=LLM_SETTINGS.litellm_chat_model_name or kwargs.get("litellm_chat_model_name", "ollama/mistral"),
            messages=messages,
            stream=kwargs.get("stream", False),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000),
            **kwargs
        )
        if system_prompt:
            logger.info(f"{LogColors.RED}system:{LogColors.END} {system_prompt}", tag="debug_llm")
        if former_messages:
            for message in former_messages:
                logger.info(f"{LogColors.CYAN}{message['role']}:{LogColors.END} {message['content']}", tag="debug_llm")
        else:
            logger.info(f"{LogColors.RED}user:{LogColors.END} {user_prompt}\n{LogColors.BLUE}resp(next row):\n{LogColors.END} {response.choices[0].message.content}", tag="debug_llm")
        
        logger.info(f"{LogColors.GREEN}Using chat model{LogColors.END} {LLM_SETTINGS.litellm_chat_model_name or kwargs.get('litellm_chat_model_name', 'ollama/mistral')}", tag="debug_llm")
        return response.choices[0].message.content
        
    def create_embedding(self, input_content_list: Union[str, List[str]],
                        *args, **kwargs) -> Union[List[Any], Any]:
        """Create embeddings using LiteLLM"""
        from litellm import embedding
        single_input = False
        if isinstance(input_content_list, str):
            input_content_list = [input_content_list]
            single_input = True
        response_list = []
        for input_content in input_content_list:
            logger.info(f"Creating embedding for: {input_content}", tag="debug_litellm_emb")
            if not isinstance(input_content, str):
                raise ValueError("Input content must be a string")
            response = embedding(
                model=LLM_SETTINGS.litellm_embedding_model_name or kwargs.get("model", "ollama/nomic-embed-text"),
                input=input_content,
                **kwargs
            )
            model_name = LLM_SETTINGS.litellm_embedding_model_name or kwargs.get("model", "ollama/nomic-embed-text")
            logger.info(f"{LogColors.GREEN}Using emb model{LogColors.END} {model_name}",tag="debug_litellm_emb")
            response_list.append(response.data[0]['embedding'])
        if single_input:
            return response_list[0]
        return response_list
    
    def build_messages_and_calculate_token(self, user_prompt: str,
                                          system_prompt: Optional[str],
                                          former_messages: Optional[List[Dict[str, Any]]] = None,
                                          shrink_multiple_break: bool = False) -> int:
        """Build messages and calculate their token count using LiteLLM"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        if former_messages:
            messages.extend(former_messages)
            
        messages.append({"role": "user", "content": user_prompt})
        
        # Calculate tokens
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(self.encoder.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
