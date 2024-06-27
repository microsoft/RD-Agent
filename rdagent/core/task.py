from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, Sequence, Tuple, TypeVar

import pandas as pd

"""
This file contains the all the data class for rdagent task.
"""


class BaseTask(ABC):
    # TODO: 把name放在这里作为主键
    # Please refer to rdagent/model_implementation/task.py for the implementation
    # I think the task version applies to the base class.
    pass


ASpecificTask = TypeVar("ASpecificTask", bound=BaseTask)


class TaskImplementation(ABC, Generic[ASpecificTask]):
    def __init__(self, target_task: ASpecificTask) -> None:
        self.target_task = target_task

    @abstractmethod
    def execute(self, data=None, config: dict = {}) -> object:
        """
        The execution of the implementation can be dynamic.

        So we may passin the data and config dynamically.
        """
        raise NotImplementedError("execute method is not implemented.")

    @abstractmethod
    def execute_desc(self):
        """
        return the description how we will execute the code in the folder.
        """
        raise NotImplementedError(f"This type of input is not supported")

    # TODO:
    # After execution, it should return some results.
    # Some evaluators will input the results and output


ASpecificTaskImp = TypeVar("ASpecificTaskImp", bound=TaskImplementation)


class ImpLoader(ABC, Generic[ASpecificTask, ASpecificTaskImp]):
    @abstractmethod
    def load(self, task: ASpecificTask) -> ASpecificTaskImp:
        raise NotImplementedError("load method is not implemented.")


class FBTaskImplementation(TaskImplementation):
    """
    File-based task implementation

    The implemented task will be a folder which contains related elements.
    - Data
    - Code Implementation
    - Output
        - After execution, it will generate the final output as file.

    A typical way to run the pipeline of FBTaskImplementation will be
    (We didn't add it as a method due to that we may pass arguments into `prepare` or `execute` based on our requirements.)

    .. code-block:: python

        def run_pipline(self, **files: str):
            self.prepare()
            self.inject_code(**files)
            self.execute()

    """

    # TODO:
    # FileBasedFactorImplementation should inherient from it.
    # Why not directly reuse FileBasedFactorImplementation.
    #   Because it has too much concerete dependencies.
    #   e.g.  dataframe, factors

    path: Optional[Path]

    @abstractmethod
    def prepare(self, *args, **kwargs):
        """
        Prepare all the files except the injected code
        - Data
        - Documentation
        - TODO: env?  Env is implicitly defined by the document?

            typical usage of `*args, **kwargs`:
                Different methods shares the same data. The data are passed by the arguments.
        """

    def inject_code(self, **files: str):
        """
        Inject the code into the folder.
        {
            "model.py": "<model code>"
        }
        """
        for k, v in files.items():
            with open(self.path / k, "w") as f:
                f.write(v)

    def get_files(self) -> list[Path]:
        """
        Get the environment description.

        To be general, we only return a list of filenames.
        How to summarize the environment is the responsibility of the TaskGenerator.
        """
        return list(self.path.iterdir())


class TestCase:
    def __init__(
        self,
        target_task: list[BaseTask] = [],
        ground_truth: list[TaskImplementation] = [],
    ):
        self.ground_truth = ground_truth
        self.target_task = target_task


class TaskLoader:
    @abstractmethod
    def load(self, *args, **kwargs) -> Sequence[BaseTask]:
        raise NotImplementedError("load method is not implemented.")
