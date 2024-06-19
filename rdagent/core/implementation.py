from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdagent.core.task import TaskImplementation


class TaskGenerator(ABC):
    @abstractmethod
    def generate(self, *args: list, **kwargs: dict) -> list[TaskImplementation]:
        error_message = "generate method is not implemented."
        raise NotImplementedError(error_message)

    @abstractmethod
    def collect_feedback(self, feedback_obj_l: list[object]) -> None:
        """
        When online evaluation.
        The previous feedbacks will be collected to support advanced factor generator

        Parameters
        ----------
        feedback_obj_l : List[object]

        """




