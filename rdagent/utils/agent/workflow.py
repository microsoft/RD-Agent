import json
from typing import Any, Callable, Type, TypeVar, Union, cast

from rdagent.core.exception import FormatError
from rdagent.log import rdagent_logger as logger

T = TypeVar("T")


def build_cls_from_json_with_retry(
    cls: Type[T],
    system_prompt: str,
    user_prompt: str,
    retry_n: int = 5,
    init_kwargs_update_func: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    **kwargs: dict,
) -> T:
    """
    Parameters
    ----------
    cls : Type[T]
        The class type to be instantiated with the response data.
    system_prompt : str
        The initial prompt provided to the system for context.
    user_prompt : str
        The prompt given by the user to guide the response generation.
    retry_n : int
        The number of attempts to retry in case of failure.
    init_kwargs_update_func : Union[Callable[[dict], dict], None]
        A function that takes the initial keyword arguments as input and returns the updated keyword arguments.
        This function can be used to modify the response data before it is used to instantiate the class.

    **kwargs
        Additional keyword arguments passed to the API call.

    Returns
    -------
    T
        An instance of the specified class type created from the response data.
    """
    from rdagent.oai.llm_utils import APIBackend  # avoid circular import

    for i in range(retry_n):
        # currently, it only handle exception caused by initial class
        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True, **kwargs  # type: ignore[arg-type]
        )
        try:
            resp_dict = json.loads(resp)
            if init_kwargs_update_func:
                resp_dict = init_kwargs_update_func(resp_dict)
            return cls(**resp_dict)
        except Exception as e:
            logger.warning(f"Attempt {i + 1}: The previous attempt didn't work due to: {e}")
            user_prompt = user_prompt + f"\n\nAttempt {i + 1}: The previous attempt didn't work due to: {e}"
    raise FormatError("Unable to produce a JSON response that meets the specified requirements.")
