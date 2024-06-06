from scripts.benchmark.config import TASK_VERSION
from rdagent.core.implementation import TaskImplementation
from typing import List
from rdagent.core.task import FactorTask, TestCase
# TODO:Need to verify the type of input data，how to deal with the gt

# (haoxue) need to check the following code, it seems that there exists task.py 
class task(object):
    def __init__(self, task_name, task_description, task_formulation, task_formulation_description, variables: dict = {}, resource: str = None):
        self.task_name = task_name
        self.task_description = task_description
        self.task_formulation = task_formulation
        self.task_formulation_description = task_formulation_description
        self.variables = variables
        self.task_resources = resource
    def task_key_adaptor(self, fname, task):
        # FIXME: we should align the code and task to make the interface simpler
        res = {"factor_name": fname}
        for k, v in task.items():
            res[
                {
                    "formulation": "factor_formulation",
                    "description": "factor_description",
                    "variable": "variables",
                }.get(k, k)
            ] = v
            res.update(
                {
                    # "factor_description": "",
                    "factor_formulation_description": "",
                },
            )
        return res

    def task_set_adaptor(self, task_set):
        res = {}
        for k, f_d in task_set.items():
            new_f_d = [self.task_key_adaptor(f, d) for f, d in f_d.items()]
            res[k] = new_f_d
        return res

    def load_all_task_json_disk(self, path):
        with open(path) as f:
            data = json.load(f)
        return self.task_set_adaptor(data)


def get_test_task_json(version: TASK_VERSION, path):
    if version == "":
        res=task().load_all_task_json_disk(path=path)
    elif version == "random":
        pass
    elif version == "Naive":
        pass
    elif version == "CoT":
        pass
    elif version == "Past":
        pass
    else:
        raise ValueError(f"Unknown version: {version}")
    return res

def load_tasks(data: dict, with_fname=False):
    # TODO: we should put these into a staticmethod as BaseEval (maybe FactorImplementTask)
    # load tasks from json
    ft_l = []
    for fname, factor_list in data.items():
        for t in factor_list:
            ft = FactorTask.from_dict(t)
            # key in factor
            if "variables" in t:
                ft.factor_formulation_description = str(t["variables"])
            if with_fname:
                ft_l.append((fname, ft))
            else:
                ft_l.append(ft)
    return ft_l

def load_eval_data(version: TASK_VERSION, path) -> List[TestCase]:
    # prepare the input data used for generation
    # The process should contain: 1. read the factor/model info, 2. Prepare and check the gt
    all_task_json = get_test_task_json(version, path)
    return load_tasks(all_task_json)

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
        target_task: FactorTask,
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
    def from_folder(task: FactorTask, path: Union[str, Path], **kwargs):
        path = Path(path)
        factor_path = (path / task.factor_name).with_suffix(".py")
        with factor_path.open("r") as f:
            code = f.read()
        return FileBasedFactorImplementation(task, code=code, **kwargs)

