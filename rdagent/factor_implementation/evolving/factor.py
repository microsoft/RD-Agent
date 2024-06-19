from __future__ import annotations

import pickle
import subprocess
import uuid
from pathlib import Path

import pandas as pd
from filelock import FileLock
from rdagent.core.evolving_framework import EvolvableSubjects
from rdagent.core.exception import (
    CodeFormatError,
    NoOutputError,
    RuntimeError,
)
from rdagent.core.log import RDAgentLog
from rdagent.core.task import (
    BaseTask,
    TaskImplementation,
)
from rdagent.factor_implementation.share_modules.factor_implementation_config import (
    FACTOR_IMPLEMENT_SETTINGS,
)
from rdagent.oai.llm_utils import md5_hash


class FactorImplementTask(BaseTask):
    def __init__(
        self,
        factor_name: str,
        factor_description: str,
        factor_formulation: str,
        factor_formulation_description: str = "",
        variables: dict | None = None,
        resource: str | None = None,
    ) -> None:
        if variables is None:
            variables = {}
        self.factor_name = factor_name
        self.factor_description = factor_description
        self.factor_formulation = factor_formulation
        self.factor_formulation_description = factor_formulation_description
        self.variables = variables
        self.factor_resources = resource

    def get_factor_information(self) -> str:
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
factor_formulation_description: {self.factor_formulation_description}"""

    @staticmethod
    def from_dict(data_dict: dict) -> FactorImplementTask:
        return FactorImplementTask(**data_dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.factor_name}]>"


class FactorEvovlingItem(EvolvableSubjects):
    """
    Intermediate item of factor implementation.
    """

    def __init__(
        self,
        target_factor_tasks: list[FactorImplementTask],
        corresponding_gt_implementations: list[TaskImplementation] | None = None,
    ) -> None:
        super().__init__()
        self.target_factor_tasks = target_factor_tasks
        self.corresponding_implementations: list[TaskImplementation] = [None for _ in target_factor_tasks]
        self.corresponding_selection: list = None
        if corresponding_gt_implementations is not None and len(
            corresponding_gt_implementations,
        ) != len(target_factor_tasks):
            self.corresponding_gt_implementations = None
            RDAgentLog().warning(
                "The length of corresponding_gt_implementations is not equal"
                " to the length of target_factor_tasks, set"
                " corresponding_gt_implementations to None",
            )
        else:
            self.corresponding_gt_implementations = corresponding_gt_implementations


class FileBasedFactorImplementation(TaskImplementation):
    """
    This class is used to implement a factor by writing the code to a file.
    Input data and output factor value are also written to files.
    """

    # TODO: (Xiao) think raising errors may get better information for processing
    FB_FROM_CACHE = "The factor value has been executed and stored in the instance variable."
    FB_EXEC_SUCCESS = "Execution succeeded without error."
    FB_CODE_NOT_SET = "code is not set."
    FB_EXECUTION_SUCCEEDED = "Execution succeeded without error."
    FB_OUTPUT_FILE_NOT_FOUND = "\nExpected output file not found."
    FB_OUTPUT_FILE_FOUND = "\nExpected output file found."

    def __init__(
        self,
        target_task: FactorImplementTask,
        code: str,
        executed_factor_value_dataframe: pd.DataFrame = None,
        raise_exception: bool = False,
    ) -> None:
        super().__init__(target_task)
        self.code = code
        self.executed_factor_value_dataframe = executed_factor_value_dataframe
        self.logger = RDAgentLog()
        self.raise_exception = raise_exception
        self.workspace_path = Path(
            FACTOR_IMPLEMENT_SETTINGS.file_based_execution_workspace,
        ) / str(uuid.uuid4())

    @staticmethod
    def link_data_to_workspace(data_path: Path | str, workspace_path: Path | str) -> None:
        data_path = Path(data_path)
        workspace_path = Path(workspace_path)
        for data_file_path in data_path.iterdir():
            workspace_data_file_path = workspace_path / data_file_path.name
            if workspace_data_file_path.exists():
                workspace_data_file_path.unlink()
            subprocess.run(
                ["ln", "-s", str(data_file_path), str(workspace_data_file_path)],
                check=False,
            )

    def execute(self, *, store_result: bool = False) -> tuple[str, pd.DataFrame]:
        """
        Execute the implementation and get the factor value by the following steps:
        1. Make the directory in workspace path
        2. Write the code to the file in the workspace path
        3. Link all the source data to the workspace path folder
        4. Execute the code
        5. Read the factor value from the output file in the workspace path folder
        Returns the execution feedback as a string and the factor value as a pandas dataframe.

        Parameters:
        store_result: If True, store the factor value in the instance variable. This feature is to be used in
        the gt implementation to avoid multiple executions on the same gt implementation.
        """
        if self.code is None:
            if self.raise_exception:
                raise CodeFormatError(self.FB_CODE_NOT_SET)
            # TODO: to make the interface compatible with previous code. I kept the original behavior.
            raise ValueError(self.FB_CODE_NOT_SET)
        with FileLock(self.workspace_path / "execution.lock"):
            if FACTOR_IMPLEMENT_SETTINGS.enable_execution_cache:
                # NOTE: cache the result for the same code
                target_file_name = md5_hash(self.code)
                cache_file_path = (
                    Path(FACTOR_IMPLEMENT_SETTINGS.implementation_execution_cache_location) / f"{target_file_name}.pkl"
                )
                Path(FACTOR_IMPLEMENT_SETTINGS.implementation_execution_cache_location).mkdir(
                    exist_ok=True, parents=True,
                )
                if cache_file_path.exists() and not self.raise_exception:
                    with cache_file_path.open("rb") as cache_file:
                        cached_res = pickle.load(cache_file)
                    if store_result and cached_res[1] is not None:
                        self.executed_factor_value_dataframe = cached_res[1]
                    return cached_res

            if self.executed_factor_value_dataframe is not None:
                return self.FB_FROM_CACHE, self.executed_factor_value_dataframe

            source_data_path = Path(
                FACTOR_IMPLEMENT_SETTINGS.file_based_execution_data_folder,
            )
            self.workspace_path.mkdir(exist_ok=True, parents=True)

            code_path = self.workspace_path / f"{self.target_task.factor_name}.py"
            code_path.write_text(self.code)

            self.link_data_to_workspace(source_data_path, self.workspace_path)

            execution_feedback = self.FB_EXECUTION_SUCCEEDED
            try:
                subprocess.check_output(
                    ["python", str(code_path)],
                    cwd=self.workspace_path,
                    stderr=subprocess.STDOUT,
                    timeout=FACTOR_IMPLEMENT_SETTINGS.file_based_execution_timeout,
                )
            except subprocess.CalledProcessError as e:
                import site

                execution_feedback = (
                    e.output.decode()
                    .replace(str(code_path.parent.absolute()), r"/path/to")
                    .replace(str(site.getsitepackages()[0]), r"/path/to/site-packages")
                )
                MAX_FEEDBACK_LENGTH = 2000
                if len(execution_feedback) > MAX_FEEDBACK_LENGTH:
                    execution_feedback = (
                        execution_feedback[:1000] + "....hidden long error message...." + execution_feedback[-1000:]
                    )
                if self.raise_exception:
                    raise RuntimeError(execution_feedback) from e
            except subprocess.TimeoutExpired as e:
                execution_feedback += f"Execution timeout error and the timeout is set to {FACTOR_IMPLEMENT_SETTINGS.file_based_execution_timeout} seconds."
                if self.raise_exception:
                    raise RuntimeError(execution_feedback) from e

            workspace_output_file_path = self.workspace_path / "result.h5"
            if not workspace_output_file_path.exists():
                execution_feedback += self.FB_OUTPUT_FILE_NOT_FOUND
                executed_factor_value_dataframe = None
                if self.raise_exception:
                    raise NoOutputError(execution_feedback)
            else:
                try:
                    executed_factor_value_dataframe = pd.read_hdf(workspace_output_file_path)
                    execution_feedback += self.FB_OUTPUT_FILE_FOUND
                except Exception as e:
                    execution_feedback += f"Error found when reading hdf file: {e}"[:1000]
                    executed_factor_value_dataframe = None

            if store_result and executed_factor_value_dataframe is not None:
                self.executed_factor_value_dataframe = executed_factor_value_dataframe

        if FACTOR_IMPLEMENT_SETTINGS.enable_execution_cache:
            with cache_file_path.open("wb") as cache_file:
                pickle.dump(
                    (execution_feedback, executed_factor_value_dataframe),
                    cache_file,
                )
        return execution_feedback, executed_factor_value_dataframe

    def __str__(self) -> str:
        # NOTE:
        # If the code cache works, the workspace will be None.
        return f"File Factor[{self.target_task.factor_name}]: {self.workspace_path}"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_folder(task: FactorImplementTask, path: str | Path, **kwargs: dict) -> FileBasedFactorImplementation:
        path = Path(path)
        factor_path = (path / task.factor_name).with_suffix(".py")
        with factor_path.open("r") as f:
            code = f.read()
        return FileBasedFactorImplementation(task, code=code, **kwargs)
