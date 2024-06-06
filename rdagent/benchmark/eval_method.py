from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from rdagent.core.conf import FactorImplementSettings
from rdagent.core.exception import ImplementRunException
from rdagent.core.task import (
    TaskImplementation,
    FactorTask,
    TestCase,
)
from rdagent.factor_implementation.evolving.evaluators import (
    FactorImplementationCorrelationEvaluator,
    FactorImplementationIndexEvaluator,
    FactorImplementationIndexFormatEvaluator,
    FactorImplementationMissingValuesEvaluator,
    FactorImplementationRowCountEvaluator,
    FactorImplementationSingleColumnEvaluator,
    FactorImplementationValuesEvaluator,
    FactorImplementationEvaluator,
)
from rdagent.core.implementation import TaskGenerator
from rdagent.utils.misc import multiprocessing_wrapper

class BaseEval:
    """
    The benchmark benchmark evaluation.
    """
    def __init__(
        self,
        evaluator_l: List[FactorImplementationEvaluator],
        test_cases: List[TestCase],
        generate_method: TaskGenerator,
        catch_eval_except: bool = True,
    ):
        """Parameters
        ----------
        test_cases : List[TestCase]
            cases to be evaluated, ground truth are included in the test cases.
        evaluator_l : List[FactorImplementationEvaluator]
            A list of evaluators to evaluate the generated code.
        catch_eval_except : bool
            If we want to debug the evaluators, we recommend to set the this parameter to True.
        """
        self.evaluator_l = evaluator_l
        self.test_cases = test_cases
        self.generate_method = generate_method
        self.catch_eval_except = catch_eval_except

    def load_cases_to_eval(
        self,
        path: Union[Path, str],
        **kwargs,
    ) -> List[TaskImplementation]:
        path = Path(path)
        fi_l = []
        for tc in self.test_cases:
            try:
                fi = FileBasedFactorImplementation.from_folder(tc.task, path, **kwargs)
                fi_l.append(fi)
            except FileNotFoundError:
                print("Fail to load test case for factor: ", tc.task.factor_name)
        return fi_l

    def eval_case(
        self,
        case_gt: TaskImplementation,
        case_gen: TaskImplementation,
    ) -> List[Union[Tuple[FactorImplementationEvaluator, object], Exception]]:
        """Parameters
        ----------
        case_gt : FactorImplementation

        case_gen : FactorImplementation


        Returns
        -------
        List[Union[Tuple[FactorImplementationEvaluator, object],Exception]]
            for each item
                If the evaluation run successfully, return the evaluate results.  Otherwise, return the exception.
        """
        eval_res = []
        for ev in self.evaluator_l:
            try:
                eval_res.append((ev, ev.evaluate(case_gt, case_gen)))
                # if the corr ev is successfully evaluated and achieve the best performance, then break
            except ImplementRunException as e:
                return e
            except Exception as e:
                # exception when evaluation
                if self.catch_eval_except:
                    eval_res.append((ev, e))
                else:
                    raise e
        return eval_res
    
class FactorImplementEval(BaseEval):
    def __init__(
        self,
        test_case: TestCase,
        method: TaskGenerator,
        test_round: int = 10,
        *args,
        **kwargs,
    ):
        # evaluator collection for online evaluation
        online_evaluator_l = [
                            FactorImplementationCorrelationEvaluator,
                            FactorImplementationIndexEvaluator,
                            FactorImplementationIndexFormatEvaluator,
                            FactorImplementationMissingValuesEvaluator,
                            FactorImplementationRowCountEvaluator,
                            FactorImplementationSingleColumnEvaluator,
                            FactorImplementationValuesEvaluator,
                         ],
        super().__init__(online_evaluator_l, test_case, method, *args, **kwargs)
        self.test_round = test_round

    def eval(self):

        gen_factor_l_all_rounds = []
        test_cases_all_rounds = []
        res = defaultdict(list)
        for _ in tqdm(range(self.test_round), desc="Rounds of Eval"):
            print("\n========================================================")
            print(f"Eval {_}-th times...")
            print("========================================================\n")
            try:
                gen_factor_l = self.generate_method.generate(self.test_cases.target_task)
            except KeyboardInterrupt:
                # TODO: Why still need to save result after KeyboardInterrupt?
                print("Manually interrupted the evaluation. Saving existing results")
                break

            if len(gen_factor_l) != len(self.test_cases):
                raise ValueError(
                    "The number of cases to eval should be equal to the number of test cases.",
                )
            gen_factor_l_all_rounds.extend(gen_factor_l)
            test_cases_all_rounds.extend(self.test_cases)

        eval_res_l = []

        eval_res_list = multiprocessing_wrapper(
            [
                (self.eval_case, (gt_case.ground_truth, gen_factor))
                for gt_case, gen_factor in zip(test_cases_all_rounds, gen_factor_l_all_rounds)
            ],
            n=FactorImplementSettings().evo_multi_proc_n,
        )

        for gt_case, eval_res, gen_factor in tqdm(zip(test_cases_all_rounds, eval_res_list, gen_factor_l_all_rounds)):
            res[gt_case.task.factor_name].append((gen_factor, eval_res))
            eval_res_l.append(eval_res)

        return res
    

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




