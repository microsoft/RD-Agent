from abc import ABC, abstractmethod

from rdagent.core.task import BaseTask, TaskImplementation


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        target_task: BaseTask,
        implementation: TaskImplementation,
        gt_implementation: TaskImplementation,
        **kwargs,
    ):
        raise NotImplementedError
