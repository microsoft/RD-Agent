from __future__ import annotations

from abc import abstractmethod
from pathlib import Path


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
    def log(self, obj: object, name: str = "", **kwargs: dict) -> str | Path:
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


class View:
    """
    Motivation:

    Display the content in the storage
    """

    # TODO: pleas fix me
    @abstractmethod
    def display(s: Storage, watch: bool = False):
        """

        Parameters
        ----------
        s : Storage

        watch : bool
            should we watch the new content and display them
        """
        ...
