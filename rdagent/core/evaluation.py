from abc import ABC, abstractmethod

from rdagent.core.experiment import Implementation, Task


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        target_task: Task,
        implementation: Implementation,
        gt_implementation: Implementation,
        **kwargs,
    ):
        raise NotImplementedError
