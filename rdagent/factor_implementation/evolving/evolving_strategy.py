from __future__ import annotations

import json
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

from jinja2 import Template

from rdagent.core.evolving_framework import EvolvingStrategy, QueriedKnowledge
from rdagent.oai.llm_utils import APIBackend
from rdagent.factor_implementation.share_modules.factor_implementation_config import (
    FactorImplementSettings,
)

from rdagent.core.task import (
    TaskImplementation,
    BaseTask,
    TestCase,
)
from rdagent.core.prompts import Prompts
from rdagent.core.evolving_framework import EvolvableSubjects
from rdagent.core.log import FinCoLog

from pathlib import Path

from rdagent.factor_implementation.evolving.scheduler import (
    RandomSelect,
    LLMSelect,
)

from rdagent.factor_implementation.share_modules.factor_implementation_utils import get_data_folder_intro
from rdagent.oai.llm_utils import APIBackend, md5_hash

from utils.misc import multiprocessing_wrapper

from rdagent.core.exception import (
    CodeFormatException,
    NoOutputException,
    RuntimeErrorException,
)

if TYPE_CHECKING:
    from scripts.factor_implementation.baselines.evolving.knowledge_management import (
        FactorImplementationQueriedKnowledge,
        FactorImplementationQueriedKnowledgeV1,
    )

implement_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")

from rdagent.core.exception import (
    CodeFormatException,
    NoOutputException,
    RuntimeErrorException,
)

import pandas as pd
import uuid
import pickle
import subprocess
from typing import Tuple, Union
from filelock import FileLock

class FactorImplementTask(BaseTask):
    def __init__(
        self,
        factor_name,
        factor_description,
        factor_formulation,
        factor_formulation_description: str = '',
        variables: dict = {},
        resource: str = None,
    ) -> None:
        self.factor_name = factor_name
        self.factor_description = factor_description
        self.factor_formulation = factor_formulation
        self.factor_formulation_description = factor_formulation_description
        self.variables = variables
        self.factor_resources = resource

    def get_factor_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
factor_formulation_description: {self.factor_formulation_description}"""

    @staticmethod
    def from_dict(dict):
        return FactorImplementTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.factor_name}]>"


class FactorEvovlingItem(EvolvableSubjects):
    """
    Intermediate item of factor implementation.
    """

    def __init__(
        self,
        target_factor_tasks: list[FactorImplementTask],
        corresponding_gt: list[TestCase] = None,
        corresponding_gt_implementations: list[TaskImplementation] = None,
    ):
        super().__init__()
        self.target_factor_tasks = target_factor_tasks
        self.corresponding_implementations: list[TaskImplementation] = []
        self.corresponding_selection: list[list] = []
        self.evolve_trace = {}
        self.corresponding_gt = corresponding_gt
        if corresponding_gt_implementations is not None and len(
            corresponding_gt_implementations,
        ) != len(target_factor_tasks):
            self.corresponding_gt_implementations = None
            FinCoLog.warning(
                "The length of corresponding_gt_implementations is not equal to the length of target_factor_tasks, set corresponding_gt_implementations to None",
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
        code,
        executed_factor_value_dataframe=None,
        raise_exception=False,
    ) -> None:
        super().__init__(target_task)
        self.code = code
        self.executed_factor_value_dataframe = executed_factor_value_dataframe
        self.logger = FinCoLog()
        self.raise_exception = raise_exception
        self.workspace_path = Path(
            FactorImplementSettings().file_based_execution_workspace,
        ) / str(uuid.uuid4())

    @staticmethod
    def link_data_to_workspace(data_path: Path, workspace_path: Path):
        data_path = Path(data_path)
        workspace_path = Path(workspace_path)
        for data_file_path in data_path.iterdir():
            workspace_data_file_path = workspace_path / data_file_path.name
            if workspace_data_file_path.exists():
                workspace_data_file_path.unlink()
            subprocess.run(
                ["ln", "-s", data_file_path, workspace_data_file_path],
                check=False,
            )

    def execute(self, store_result: bool = False) -> Tuple[str, pd.DataFrame]:
        """
        execute the implementation and get the factor value by the following steps:
        1. make the directory in workspace path
        2. write the code to the file in the workspace path
        3. link all the source data to the workspace path folder
        4. execute the code
        5. read the factor value from the output file in the workspace path folder
        returns the execution feedback as a string and the factor value as a pandas dataframe

        parameters:
        store_result: if True, store the factor value in the instance variable, this feature is to be used in the gt implementation to avoid multiple execution on the same gt implementation
        """
        if self.code is None:
            if self.raise_exception:
                raise CodeFormatException(self.FB_CODE_NOT_SET)
            else:
                # TODO: to make the interface compatible with previous code. I kept the original behavior.
                raise ValueError(self.FB_CODE_NOT_SET)
        with FileLock(self.workspace_path / "execution.lock"):
            (Path.cwd() / "git_ignore_folder" / "factor_implementation_execution_cache").mkdir(
                exist_ok=True, parents=True
            )
            if FactorImplementSettings().enable_execution_cache:
                # NOTE: cache the result for the same code
                target_file_name = md5_hash(self.code)
                cache_file_path = (
                    Path.cwd()
                    / "git_ignore_folder"
                    / "factor_implementation_execution_cache"
                    / f"{target_file_name}.pkl"
                )
                if cache_file_path.exists() and not self.raise_exception:
                    cached_res = pickle.load(open(cache_file_path, "rb"))
                    if store_result and cached_res[1] is not None:
                        self.executed_factor_value_dataframe = cached_res[1]
                    return cached_res

            if self.executed_factor_value_dataframe is not None:
                return self.FB_FROM_CACHE, self.executed_factor_value_dataframe

            source_data_path = Path(
                FactorImplementSettings().file_based_execution_data_folder,
            )
            self.workspace_path.mkdir(exist_ok=True, parents=True)

            code_path = self.workspace_path / f"{self.target_task.factor_name}.py"
            code_path.write_text(self.code)

            self.link_data_to_workspace(source_data_path, self.workspace_path)

            execution_feedback = self.FB_EXECUTION_SUCCEEDED
            try:
                subprocess.check_output(
                    f"python {code_path}",
                    shell=True,
                    cwd=self.workspace_path,
                    stderr=subprocess.STDOUT,
                    timeout=FactorImplementSettings().file_based_execution_timeout,
                )
            except subprocess.CalledProcessError as e:
                import site

                execution_feedback = (
                    e.output.decode()
                    .replace(str(code_path.parent.absolute()), r"/path/to")
                    .replace(str(site.getsitepackages()[0]), r"/path/to/site-packages")
                )
                if len(execution_feedback) > 2000:
                    execution_feedback = (
                        execution_feedback[:1000] + "....hidden long error message...." + execution_feedback[-1000:]
                    )
                if self.raise_exception:
                    raise RuntimeErrorException(execution_feedback)
            except subprocess.TimeoutExpired:
                execution_feedback += f"Execution timeout error and the timeout is set to {FactorImplementSettings().file_based_execution_timeout} seconds."
                if self.raise_exception:
                    raise RuntimeErrorException(execution_feedback)

            workspace_output_file_path = self.workspace_path / "result.h5"
            if not workspace_output_file_path.exists():
                execution_feedback += self.FB_OUTPUT_FILE_NOT_FOUND
                executed_factor_value_dataframe = None
                if self.raise_exception:
                    raise NoOutputException(execution_feedback)
            else:
                try:
                    executed_factor_value_dataframe = pd.read_hdf(workspace_output_file_path)
                    execution_feedback += self.FB_OUTPUT_FILE_FOUND
                except Exception as e:
                    execution_feedback += f"Error found when reading hdf file: {e}"[:1000]
                    executed_factor_value_dataframe = None

            if store_result and executed_factor_value_dataframe is not None:
                self.executed_factor_value_dataframe = executed_factor_value_dataframe

        if FactorImplementSettings().enable_execution_cache:
            pickle.dump(
                (execution_feedback, executed_factor_value_dataframe),
                open(cache_file_path, "wb"),
            )
        return execution_feedback, executed_factor_value_dataframe

    def __str__(self) -> str:
        # NOTE:
        # If the code cache works, the workspace will be None.
        return f"File Factor[{self.target_task.factor_name}]: {self.workspace_path}"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_folder(task: FactorImplementTask, path: Union[str, Path], **kwargs):
        path = Path(path)
        factor_path = (path / task.factor_name).with_suffix(".py")
        with factor_path.open("r") as f:
            code = f.read()
        return FileBasedFactorImplementation(task, code=code, **kwargs)

class MultiProcessEvolvingStrategy(EvolvingStrategy):
    @abstractmethod
    def implement_one_factor(
        self,
        target_task: FactorImplementTask,
        queried_knowledge: QueriedKnowledge = None,
    ) -> TaskImplementation:
        raise NotImplementedError

    def evolve(
        self,
        *,
        evo: FactorEvovlingItem,
        queried_knowledge: FactorImplementationQueriedKnowledge | None = None,
        **kwargs,
    ) -> FactorEvovlingItem:
        self.num_loop += 1
        new_evo = deepcopy(evo)
        new_evo.corresponding_implementations = [None for _ in new_evo.target_factor_tasks]

        # 1.找出需要evolve的factor
        to_be_finished_task_index = []
        for index, target_factor_task in enumerate(new_evo.target_factor_tasks):
            target_factor_task_desc = target_factor_task.get_factor_information()
            if target_factor_task_desc in queried_knowledge.success_task_to_knowledge_dict:
                new_evo.corresponding_implementations[index] = queried_knowledge.success_task_to_knowledge_dict[
                    target_factor_task_desc
                ].implementation
            elif (
                target_factor_task_desc not in queried_knowledge.success_task_to_knowledge_dict
                and target_factor_task_desc not in queried_knowledge.failed_task_info_set
            ):
                to_be_finished_task_index.append(index)

        # 2. 选择selection方法
        # if the number of factors to be implemented is larger than the limit, we need to select some of them
        if FactorImplementSettings().implementation_factors_per_round < len(to_be_finished_task_index):
            # if the number of loops is equal to the select_loop, we need to select some of them
            if FactorImplementSettings().select_method == "random":
                to_be_finished_task_index = RandomSelect(
                    to_be_finished_task_index,
                    FactorImplementSettings().implementation_factors_per_round
                )

            if FactorImplementSettings().select_method == "scheduler":
                to_be_finished_task_index = LLMSelect(
                    to_be_finished_task_index,
                    FactorImplementSettings().implementation_factors_per_round,
                    new_evo,
                    queried_knowledge.all_former_traces,
                )
        
        result = multiprocessing_wrapper(
            [
                (self.implement_one_factor, (new_evo.target_factor_tasks[target_index], queried_knowledge))
                for target_index in to_be_finished_task_index
            ],
            n=FactorImplementSettings().evo_multi_proc_n,
        )

        for index, target_index in enumerate(to_be_finished_task_index):
            new_evo.corresponding_implementations[target_index] = result[index]
            if result[index].target_task.factor_name in new_evo.evolve_trace:
                new_evo.evolve_trace[result[index].target_task.factor_name].append(
                    result[index]
                )
            else:
                new_evo.evolve_trace[result[index].target_task.factor_name] = [result[index]]

        new_evo.corresponding_selection.append(to_be_finished_task_index)

        return new_evo


class FactorEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_factor(
        self,
        target_task: FactorImplementTask,
        queried_knowledge: FactorImplementationQueriedKnowledgeV1 = None,
    ) -> TaskImplementation:
        factor_information_str = target_task.get_factor_information()

        if queried_knowledge is not None and factor_information_str in queried_knowledge.success_task_to_knowledge_dict:
            return queried_knowledge.success_task_to_knowledge_dict[factor_information_str].implementation
        elif queried_knowledge is not None and factor_information_str in queried_knowledge.failed_task_info_set:
            return None
        else:
            queried_similar_successful_knowledge = (
                queried_knowledge.working_task_to_similar_successful_knowledge_dict[factor_information_str]
                if queried_knowledge is not None
                else []
            )
            queried_former_failed_knowledge = (
                queried_knowledge.working_task_to_former_failed_knowledge_dict[factor_information_str]
                if queried_knowledge is not None
                else []
            )

            queried_former_failed_knowledge_to_render = queried_former_failed_knowledge

            system_prompt = Template(
                implement_prompts["evolving_strategy_factor_implementation_v1_system"],
            ).render(
                data_info=get_data_folder_intro(),
                queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
            )
            session = APIBackend(use_chat_cache=False).build_chat_session(
                session_system_prompt=system_prompt,
            )

            queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
            while True:
                user_prompt = (
                    Template(
                        implement_prompts["evolving_strategy_factor_implementation_v1_user"],
                    )
                    .render(
                        factor_information_str=factor_information_str,
                        queried_similar_successful_knowledge=queried_similar_successful_knowledge_to_render,
                    )
                    .strip("\n")
                )
                if (
                    session.build_chat_completion_message_and_calculate_token(
                        user_prompt,
                    )
                    < FactorImplementSettings().chat_token_limit
                ):
                    break
                elif len(queried_former_failed_knowledge_to_render) > 1:
                    queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
                elif len(queried_similar_successful_knowledge_to_render) > 1:
                    queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge_to_render[1:]

            code = json.loads(
                session.build_chat_completion(
                    user_prompt=user_prompt,
                    json_mode=True,
                ),
            )["code"]
            # ast.parse(code)
            factor_implementation = FileBasedFactorImplementation(
                target_task,
                code,
            )

            return factor_implementation

class FactorEvolvingStrategyWithGraph(MultiProcessEvolvingStrategy):
    def __init__(self) -> None:
        self.num_loop = 0
        self.haveSelected = False

    def implement_one_factor(
        self,
        target_task: FactorImplementTask,
        queried_knowledge,
    ) -> TaskImplementation:
        error_summary = FactorImplementSettings().v2_error_summary
        # 1. 提取因子的背景信息
        target_factor_task_information = target_task.get_factor_information()

        # 2. 检查该因子是否需要继续做（是否已经作对，是否做错太多）
        if (
            queried_knowledge is not None
            and target_factor_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_factor_task_information].implementation
        elif queried_knowledge is not None and target_factor_task_information in queried_knowledge.failed_task_info_set:
            return None
        else:

            # 3. 取出knowledge里面的经验数据（similar success、similar error、former_trace）
            queried_similar_component_knowledge = (
                queried_knowledge.component_with_success_task[target_factor_task_information]
                if queried_knowledge is not None
                else []
            )  # A list, [success task implement knowledge]

            queried_similar_error_knowledge = (
                queried_knowledge.error_with_success_task[target_factor_task_information]
                if queried_knowledge is not None
                else {}
            )  # A dict, {{error_type:[[error_imp_knowledge, success_imp_knowledge],...]},...}

            queried_former_failed_knowledge = (
                queried_knowledge.former_traces[target_factor_task_information] if queried_knowledge is not None else []
            )

            queried_former_failed_knowledge_to_render = queried_former_failed_knowledge

            system_prompt = Template(
                implement_prompts["evolving_strategy_factor_implementation_v1_system"],
            ).render(
                data_info=get_data_folder_intro(),
                queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
            )

            session = APIBackend(use_chat_cache=False).build_chat_session(
                session_system_prompt=system_prompt,
            )

            queried_similar_component_knowledge_to_render = queried_similar_component_knowledge
            queried_similar_error_knowledge_to_render = queried_similar_error_knowledge
            error_summary_critics = ""
            # 动态地防止prompt超长
            while True:
                # 总结error（可选）
                if (
                    error_summary
                    and len(queried_similar_error_knowledge_to_render) != 0
                    and len(queried_former_failed_knowledge_to_render) != 0
                ):
                    
                    error_summary_system_prompt = (
                        Template(implement_prompts["evolving_strategy_error_summary_v2_system"])
                        .render(
                            factor_information_str=target_factor_task_information,
                            code_and_feedback=queried_former_failed_knowledge_to_render[
                                -1
                            ].get_implementation_and_feedback_str(),
                        )
                        .strip("\n")
                    )
                    session_summary = APIBackend(use_chat_cache=False).build_chat_session(
                        session_system_prompt=error_summary_system_prompt,
                    )
                    while True:
                        error_summary_user_prompt = (
                            Template(implement_prompts["evolving_strategy_error_summary_v2_user"])
                            .render(
                                queried_similar_component_knowledge=queried_similar_component_knowledge_to_render,
                            )
                            .strip("\n")
                        )
                        if (
                            session_summary.build_chat_completion_message_and_calculate_token(error_summary_user_prompt)
                            < FactorImplementSettings().chat_token_limit
                        ):
                            break
                        elif len(queried_similar_error_knowledge_to_render) > 0:
                            queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
                    error_summary_critics = session_summary.build_chat_completion(
                        user_prompt=error_summary_user_prompt,
                        json_mode=False,
                    )
                # 构建user_prompt。开始写代码
                user_prompt = (
                    Template(
                        implement_prompts["evolving_strategy_factor_implementation_v2_user"],
                    )
                    .render(
                        factor_information_str=target_factor_task_information,
                        queried_similar_component_knowledge=queried_similar_component_knowledge_to_render,
                        queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
                        error_summary=error_summary,
                        error_summary_critics=error_summary_critics,
                    )
                    .strip("\n")
                )
                if (
                    session.build_chat_completion_message_and_calculate_token(
                        user_prompt,
                    )
                    < FactorImplementSettings().chat_token_limit
                ):
                    break
                elif len(queried_former_failed_knowledge_to_render) > 1:
                    queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
                elif len(queried_similar_component_knowledge_to_render) > len(
                    queried_similar_error_knowledge_to_render,
                ):
                    queried_similar_component_knowledge_to_render = queried_similar_component_knowledge_to_render[:-1]
                elif len(queried_similar_error_knowledge_to_render) > 0:
                    queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]

            response = session.build_chat_completion(
                user_prompt=user_prompt,
                json_mode=True,
            )
            code = json.loads(response)["code"]
            factor_implementation = FileBasedFactorImplementation(target_task, code)
            return factor_implementation
