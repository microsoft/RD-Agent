from typing import Any, Optional, Union, List, Dict
import uuid
from rdagent.oai.backend.base import APIBackend
from litellm import completion, acompletion
import tiktoken
import os

class LiteLLMAPIBackend(APIBackend):
    """LiteLLM implementation of APIBackend interface"""
    
    def __init__(self, **kwargs):
        super().__init__()
        # Set up any required LiteLLM configurations
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
    def build_chat_session(self, conversation_id: Optional[str] = None,
                         session_system_prompt: Optional[str] = None) -> Any:
        """Create a new chat session using LiteLLM"""
        return {
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "system_prompt": session_system_prompt,
            "messages": []
        }
        
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
            model=kwargs.get("model", "gpt-3.5-turbo"),
            messages=messages,
            stream=kwargs.get("stream", False),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000),
            **kwargs
        )
        
        return response.choices[0].message.content
        
    def create_embedding(self, input_content: Union[str, List[str]],
                        *args, **kwargs) -> Union[List[Any], Any]:
        """Create embeddings using LiteLLM"""
        from litellm import embedding
        
        if isinstance(input_content, str):
            input_content = [input_content]
            
        response = embedding(
            model=kwargs.get("model", "text-embedding-ada-002"),
            input=input_content,
            **kwargs
        )
        
        if isinstance(input_content, str):
            return response.data[0].embedding
        return [item.embedding for item in response.data]
        
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
