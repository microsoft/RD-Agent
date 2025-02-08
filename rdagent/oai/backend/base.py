from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union


class APIBackend(ABC):
    """Abstract base class for LLM API backends"""

    @abstractmethod
    def build_chat_session(
        self, conversation_id: Optional[str] = None, session_system_prompt: Optional[str] = None
    ) -> Any:
        """Create a new chat session"""
        pass

    @abstractmethod
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
        """Build messages and get chat completion"""
        pass

    @abstractmethod
    def create_embedding(
        self, input_content: Union[str, List[str]], *args: Any, **kwargs: Any
    ) -> Union[List[Any], Any]:
        """Create embeddings for input text"""
        pass

    @abstractmethod
    def build_messages_and_calculate_token(
        self,
        user_prompt: str,
        system_prompt: Optional[str],
        former_messages: Optional[List[Dict[str, Any]]] = None,
        *,
        shrink_multiple_break: bool = False,
    ) -> int:
        """Build messages and calculate their token count"""
        pass


# TODO: seperate cache layer. try to be tranparent.
class CachedAPIBackend(APIBackend):
    ...
    # @abstractmethod
    # def none_cache_function ...
