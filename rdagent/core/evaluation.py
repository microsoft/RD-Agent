"""
It is expected to be shared among different frameworks.
"""

from abc import ABC, abstractmethod


class Feedback:
    """
    Design Principle:
        It will be more like a **dataclass**.
        The building process of feedback will should be in evaluator
    """

    def finished(self) -> bool:
        """
        In some implementations, tasks may fail multiple times, leading agents to skip the implementation.
        So both skip and success indicate the task is finished.
        """
        return self.__bool__()

    def __bool__(self) -> bool:
        return True


class EvaluableObj:
    """
    A set of information that is evaluable. Following things can be included.
    - Task
    - Solution
    - Ground Truth
    """


class Evaluator(ABC):
    """
    Design Principle:

        It should cover the building process of feedback from raw information.
            Typically the building of feedback will be two phases.
            1. raw information including stdout & workspace  (feedback itself will handle this)
            2. advanced/summarized feedback information. (evaluate will handle this)
    """

    @abstractmethod
    def evaluate(
        self,
        eo: EvaluableObj,
    ) -> Feedback:
        raise NotImplementedError
