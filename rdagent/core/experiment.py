from __future__ import annotations

import io
import os
import platform
import re
import shutil
import typing
import uuid
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, List, TypeVar

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evaluation import Feedback

if TYPE_CHECKING:
    from rdagent.utils.env import EnvResult


if typing.TYPE_CHECKING:
    from rdagent.core.proposal import Hypothesis
    from rdagent.utils.env import Env

"""
This file contains the all the class about organizing the task in RD-Agent.
"""


class AbsTask(ABC):
    def __init__(self, name: str, version: int = 1) -> None:
        """
        The version of the task, default is 1
        Because qlib tasks execution and kaggle tasks execution are different, we need to distinguish them.
        TODO: We may align them in the future.
        """
        self.version = version
        self.name = name

    @abstractmethod
    def get_task_information(self) -> str:
        """
        Get the task information string to build the unique key
        """


class UserInstructions(List[str]):
    def __str__(self) -> str:
        if self:
            return ("\nUser Instructions (Top priority!):\n" + "\n".join(f"- {ui}" for ui in self)) if self else ""
        else:
            return ""


class Task(AbsTask):
    def __init__(
        self,
        name: str,
        version: int = 1,
        description: str = "",
        user_instructions: UserInstructions | None = None,
    ) -> None:
        super().__init__(name, version)
        self.description = description
        self.user_instructions = user_instructions

    def get_task_information(self) -> str:
        return f"Task Name: {self.name}\nDescription: {self.description}{str(self.user_instructions)}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"


ASpecificTask = TypeVar("ASpecificTask", bound=Task)
ASpecificFeedback = TypeVar("ASpecificFeedback", bound=Feedback)


@dataclass
class RunningInfo:
    result: object = None  # The result of the experiment, can be different types in different scenarios.
    running_time: float | None = None


class Workspace(ABC, Generic[ASpecificTask, ASpecificFeedback]):
    """
    A workspace is a place to store the task implementation. It evolves as the developer implements the task.
    To get a snapshot of the workspace, make sure call `copy` to get a copy of the workspace.
    """

    def __init__(self, target_task: ASpecificTask | None = None) -> None:
        self.target_task: ASpecificTask | None = target_task
        self.feedback: ASpecificFeedback | None = None
        self.running_info: RunningInfo = RunningInfo()

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> object | None:
        error_message = "execute method is not implemented."
        raise NotImplementedError(error_message)

    @abstractmethod
    def copy(self) -> Workspace:
        error_message = "copy method is not implemented."
        raise NotImplementedError(error_message)

    @property
    @abstractmethod
    def all_codes(self) -> str:
        """
        Get all the code files in the workspace as a single string.
        """

    # when the workspace is mutable inplace, provide support for creating checkpoints and recovering.
    @abstractmethod
    def create_ws_ckp(self) -> None:
        """
        Create an in-memory checkpoint of the workspace so it can be restored later.
        """

    @abstractmethod
    def recover_ws_ckp(self) -> None:
        """
        Restore the workspace from the checkpoint created by :py:meth:`create_ws_ckp`.
        """


ASpecificWS = TypeVar("ASpecificWS", bound=Workspace)


class WsLoader(ABC, Generic[ASpecificTask, ASpecificWS]):
    @abstractmethod
    def load(self, task: ASpecificTask) -> ASpecificWS:
        error_message = "load method is not implemented."
        raise NotImplementedError(error_message)


class FBWorkspace(Workspace):
    """
    File-based task workspace

    The implemented task will be a folder which contains related elements.
    - Data
    - Code Workspace
    - Output
        - After execution, it will generate the final output as file.

    A typical way to run the pipeline of FBWorkspace will be:
    (We didn't add it as a method due to that we may pass arguments into
    `prepare` or `execute` based on our requirements.)

    .. code-block:: python

        def run_pipeline(self, **files: str):
            self.prepare()
            self.inject_files(**files)
            self.execute()

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.file_dict: dict[str, Any] = (
            {}
        )  # The code injected into the folder, store them in the variable to reproduce the former result
        self.workspace_path: Path = RD_AGENT_SETTINGS.workspace_path / uuid.uuid4().hex
        self.ws_ckp: bytes | None = None  # In-memory checkpoint data created by ``create_ws_ckp``.
        self.change_summary: str | None = None  # The change from the previous version of workspace

    @staticmethod
    def _format_code_dict(code_dict: dict[str, str]) -> str:
        """
        Helper function to format the code dictionary into a string.
        """
        code_string = ""
        for file_name in sorted(code_dict.keys()):
            code_string += f"\nFile Path: {file_name}\n```\n{code_dict[file_name]}\n```"
        return code_string

    @property
    def all_codes(self) -> str:
        """
        Get all the code files in the workspace as a single string, excluding test files.
        """
        filtered_dict = {k: v for k, v in self.file_dict.items() if k.endswith(".py") and "test" not in k}
        return self._format_code_dict(filtered_dict)

    def get_codes(self, pattern: str) -> str:
        """
        Get code files matching a specific pattern as a single string, excluding test files.
        """
        filtered_dict = {
            k: v for k, v in self.file_dict.items() if re.search(pattern, k) and k.endswith(".py") and "test" not in k
        }
        return self._format_code_dict(filtered_dict)

    def prepare(self) -> None:
        """
        Prepare the workspace except the injected code
        - Data
        - Documentation
            typical usage of `*args, **kwargs`:
                Different methods shares the same data. The data are passed by the arguments.
        """
        self.workspace_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def link_all_files_in_folder_to_workspace(data_path: Path, workspace_path: Path) -> None:
        data_path = Path(data_path).absolute()  # in case of relative path that will be invalid when we change cwd.
        workspace_path = Path(workspace_path)
        for data_file_path in data_path.iterdir():
            workspace_data_file_path = workspace_path / data_file_path.name
            if workspace_data_file_path.exists():
                workspace_data_file_path.unlink()
            if platform.system() in ("Linux", "Darwin"):
                workspace_data_file_path.symlink_to(data_file_path)
            if platform.system() == "Windows":
                os.link(data_file_path, workspace_data_file_path)

    DEL_KEY = "__DEL__"

    def inject_files(self, **files: str) -> None:
        """
        Inject the code into the folder.
        {
            <file name1>: <code>,  // indicate writing <code> into <file name>
                          (create new file or replace existing file)
            <file name2>: "__DEL__"  // indicate removing file name2. When we want to replace a file to a new one,
                          we usually use this
        }
        """
        self.prepare()
        for k, v in files.items():
            target_file_path = self.workspace_path / k  # Define target_file_path before using it
            if v == self.DEL_KEY:  # Use self.DEL_KEY to access the class variable
                if target_file_path.exists():
                    target_file_path.unlink()  # Unlink the file if it exists
                self.file_dict.pop(k, None)  # Safely remove the key from file_dict
            else:
                self.file_dict[k] = v
                target_file_path.parent.mkdir(parents=True, exist_ok=True)
                target_file_path.write_text(v)

    def get_files(self) -> list[Path]:
        """
        Get the environment description.

        To be general, we only return a list of filenames.
        How to summarize the environment is the responsibility of the Developer.
        """
        return list(self.workspace_path.iterdir())

    def inject_code_from_folder(self, folder_path: Path) -> None:
        """
        Load the workspace from the folder
        """
        for file_path in folder_path.rglob("*"):
            if file_path.suffix in (".py", ".yaml", ".md"):
                relative_path = file_path.relative_to(folder_path)
                self.inject_files(**{str(relative_path): file_path.read_text()})

    def inject_code_from_file_dict(self, workspace: FBWorkspace) -> None:
        """
        Load the workspace from the file_dict
        """
        for name, code in workspace.file_dict.items():
            self.inject_files(**{name: code})

    def copy(self) -> FBWorkspace:
        """
        copy the workspace from the original one
        """
        return deepcopy(self)

    def clear(self) -> None:
        """
        Clear the workspace
        """
        shutil.rmtree(self.workspace_path, ignore_errors=True)
        self.file_dict = {}

    def before_execute(self) -> None:
        """
        Before executing the code, we need to prepare the workspace and inject code into the workspace.
        """
        self.prepare()
        self.inject_files(**self.file_dict)

    def execute(self, env: Env, entry: str) -> str:
        """
        Before each execution, make sure to prepare and inject code.
        """
        result = self.run(env, entry)
        return result.get_truncated_stdout()  # NOTE: truncating just for aligning with the old code.

    def run(self, env: Env, entry: str) -> EnvResult:
        """
        Execute the code in the environment and return an EnvResult object (stdout, exit_code, running_time).

        Before each execution, make sure to prepare and inject code.
        """
        self.prepare()
        self.inject_files(**self.file_dict)
        return env.run(entry, str(self.workspace_path), env={"PYTHONPATH": "./"})

    def create_ws_ckp(self) -> None:
        """
        Zip the contents of ``workspace_path`` and persist the archive on
        ``self.ws_ckp`` for later restoration via :py:meth:`recover_ws_ckp`.
        """
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in self.workspace_path.rglob("*"):
                # Only include regular files up to 100 KB so that the checkpoint
                # remains lightweight. Larger files (for example, datasets) are
                # expected to be recreated or mounted separately.
                if file_path.is_symlink():
                    # Preserve symbolic links within the archive
                    zi = zipfile.ZipInfo(str(file_path.relative_to(self.workspace_path)))
                    zi.create_system = 3  # indicates Unix
                    zi.external_attr = 0o120777 << 16  # symlink file type + 0777 perms
                    zf.writestr(zi, str(file_path.readlink()))
                elif file_path.is_file():
                    size_limit = RD_AGENT_SETTINGS.workspace_ckp_size_limit
                    if (
                        RD_AGENT_SETTINGS.workspace_ckp_white_list_names is not None
                        and file_path.name in RD_AGENT_SETTINGS.workspace_ckp_white_list_names
                    ) or (size_limit <= 0 or file_path.stat().st_size <= size_limit):
                        zf.write(file_path, file_path.relative_to(self.workspace_path))
        self.ws_ckp = buf.getvalue()

    def recover_ws_ckp(self) -> None:
        """
        Restore the workspace directory from the in-memory checkpoint created by
        :py:meth:`create_ws_ckp`.
        """
        if self.ws_ckp is None:
            msg = "Workspace checkpoint doesn't exist. Call `create_ws_ckp` first."
            raise RuntimeError(msg)
        shutil.rmtree(self.workspace_path, ignore_errors=True)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO(self.ws_ckp)
        with zipfile.ZipFile(buf, "r") as zf:
            for info in zf.infolist():
                dest_path = self.workspace_path / info.filename
                # File type bits (upper 4) are in high 16 bits of external_attr
                mode = (info.external_attr >> 16) & 0o170000
                symlink_mode = 0o120000  # Constant for symlink file type in Unix
                if mode == symlink_mode:  # Symlink
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    link_target = zf.read(info).decode()
                    dest_path.symlink_to(link_target)
                elif info.is_dir():
                    dest_path.mkdir(parents=True, exist_ok=True)
                else:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with dest_path.open("wb") as f:
                        f.write(zf.read(info))
        # NOTE: very important to reduce the size of the object
        self.ws_ckp = None

    def __str__(self) -> str:
        return f"Workspace[{self.workspace_path=}" + (
            "]" if self.target_task is None else f",{self.target_task.name=}]"
        )


ASpecificWSForExperiment = TypeVar("ASpecificWSForExperiment", bound=Workspace)
ASpecificWSForSubTasks = TypeVar("ASpecificWSForSubTasks", bound=Workspace)


class ExperimentPlan(dict[str, Any]):
    """
    A plan for the experiment, which is a dictionary that contains the plan to each stage.
    """


class Experiment(
    ABC,
    Generic[ASpecificTask, ASpecificWSForExperiment, ASpecificWSForSubTasks],
):
    """
    The experiment is a sequence of tasks and the implementations of the tasks after generated by the Developer.
    """

    def __init__(
        self,
        sub_tasks: Sequence[ASpecificTask],
        based_experiments: Sequence[ASpecificWSForExperiment] = [],
        hypothesis: Hypothesis | None = None,
    ) -> None:
        self.hypothesis: Hypothesis | None = hypothesis  # Experiment is optionally generated by hypothesis
        self.sub_tasks: Sequence[ASpecificTask] = sub_tasks
        # None means
        # - initialization placeholder  before implementation
        # - the developer actively skip the task;
        self.sub_workspace_list: list[ASpecificWSForSubTasks | None] = [None] * len(self.sub_tasks)
        # TODO:
        # It will be used in runner in history
        # If we implement the whole workflow, we don't have to use it, then we remove it.
        self.based_experiments: Sequence[ASpecificWSForExperiment] = based_experiments

        self.experiment_workspace: ASpecificWSForExperiment | None = None

        # The experiment may be developed by different developers.
        # Last feedback is used to propagate info to the next developer.
        # Life cycle:
        # - Developer assigns feedback for next component;
        # - Workflow control clears feedback.
        self.prop_dev_feedback: Feedback | None = None

        # TODO: (xiao) I think this is too concrete; we should move it into
        # NOTE: Assumption
        # - only runner will assign this variable
        # - We will always create a new Experiment without copying previous results when we goto the next new loop.
        self.running_info = RunningInfo()
        self.sub_results: dict[str, float] = (
            {}
        )  # TODO: in Kaggle, now sub results are all saved in self.result, remove this in the future.

        # For parallel multi-trace support
        self.local_selection: tuple[int, ...] | None = None
        self.plan: ExperimentPlan | None = (
            None  # To store the planning information for this experiment, should be generated inside exp_gen.gen
        )
        self.user_instructions: UserInstructions | None = None  # To store the user instructions for this experiment

    def set_user_instructions(self, user_instructions: UserInstructions | None) -> None:
        if user_instructions is None:
            return
        if not isinstance(user_instructions, UserInstructions) and isinstance(user_instructions, list):
            user_instructions = UserInstructions(user_instructions)
        self.user_instructions = user_instructions
        for ws in self.sub_workspace_list:
            if ws is not None:
                ws.target_task.user_instructions = user_instructions  # type: ignore[union-attr]
        for task in self.sub_tasks:
            task.user_instructions = user_instructions
        if self.experiment_workspace is not None and self.experiment_workspace.target_task is not None:
            self.experiment_workspace.target_task.user_instructions = user_instructions

    @property
    def result(self) -> object:
        return self.running_info.result

    @result.setter
    def result(self, value: object) -> None:
        self.running_info.result = value

    # when the workspace is mutable inplace, provide support for creating checkpoints and recovering.
    def create_ws_ckp(self) -> None:
        if self.experiment_workspace is not None:
            self.experiment_workspace.create_ws_ckp()
        for ws in self.sub_workspace_list:
            if ws is not None:
                ws.create_ws_ckp()

    def recover_ws_ckp(self) -> None:
        if self.experiment_workspace is not None:
            self.experiment_workspace.recover_ws_ckp()
        for ws in self.sub_workspace_list:
            if ws is not None:
                try:
                    ws.recover_ws_ckp()
                except RuntimeError:
                    # the FBWorkspace is shared between experiment_workspace and sub_workspace_list,
                    # so recover_ws_ckp will raise RuntimeError if a workspace is recovered twice.
                    print("recover_ws_ckp failed due to one workspace is recovered twice.")


ASpecificExp = TypeVar("ASpecificExp", bound=Experiment)
ASpecificPlan = TypeVar("ASpecificPlan", bound=ExperimentPlan)

TaskOrExperiment = TypeVar("TaskOrExperiment", Task, Experiment)


class Loader(ABC, Generic[TaskOrExperiment]):
    @abstractmethod
    def load(self, *args: Any, **kwargs: Any) -> TaskOrExperiment:
        err_msg = "load method is not implemented."
        raise NotImplementedError(err_msg)
