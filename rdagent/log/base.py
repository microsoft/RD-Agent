from __future__ import annotations

from abc import abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union


@dataclass
class Message:
    """The info unit of the storage"""

    tag: str  # namespace like like a.b.c
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]  # The level of the logging
    timestamp: datetime  # The time when the message is generated
    caller: Optional[
        str
    ]  # The caller of the logging like `rdagent.oai.llm_utils:_create_chat_completion_inner_function:55`(file:func:line)
    pid_trace: Optional[str]  # The process id trace;  A-B-C represents A create B, B create C
    content: object  # The content


class Storage:
    """
    Basic storage to support saving objects;

    # Usage:

    The storage has mainly two kind of users:
    - The logging end: you can choose any of the following method to use the object
        - We can use it directly with the native logging storage
        - We can use it with other logging tools; For example, serve as a handler for loggers
    - The view end:
        - Mainly for the subclass of `logging.base.View`
        - It should provide two kind of ways to provide content
            - offline content provision.
            - online content preovision.
    """

    @abstractmethod
    def log(
        self,
        obj: object,
        name: str = "",
        save_type: Literal["json", "text", "pkl"] = "text",
        timestamp: datetime | None = None,
        **kwargs: dict,
    ) -> str | Path:
        """

        Parameters
        ----------
        obj : object
            The object for logging.
        name : str
            The name of the object.  For example "a.b.c"
            We may log a lot of objects to a same name

        Returns
        -------
        str | Path
            The storage identifier of the object.
        """
        ...

    @abstractmethod
    def iter_msg(self, watch: bool = False) -> Generator[Message, None, None]:
        """
        Parameters
        ----------
        watch : bool
            should we watch the new content and display them
        """
        ...


class View:
    """
    Motivation:

    Display the content in the storage
    """

    # TODO: pleas fix me
    @abstractmethod
    def display(self, s: Storage, watch: bool = False) -> None:
        """

        Parameters
        ----------
        s : Storage

        watch : bool
            should we watch the new content and display them
        """
        ...
