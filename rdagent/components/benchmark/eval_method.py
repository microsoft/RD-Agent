from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Union

from tqdm import tqdm

from rdagent.components.coder.factor_coder.config import FACTOR_IMPLEMENT_SETTINGS
from rdagent.components.coder.factor_coder.CoSTEER.evaluators import (
    FactorCorrelationEvaluator,
    FactorEqualValueCountEvaluator,
    FactorEvaluator,
    FactorIndexEvaluator,
    FactorMissingValuesEvaluator,
    FactorOutputFormatEvaluator,
    FactorRowCountEvaluator,
    FactorSingleColumnEvaluator,
)
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import CoderError
from rdagent.core.experiment import Task, Workspace
from rdagent.core.utils import multiprocessing_wrapper


class TestCase:
    def __init__(
        self,
        target_task: list[Task] = [],
        ground_truth: list[Workspace] = [],
    ):
        self.ground_truth = ground_truth
        self.target_task = target_task


class BaseEval:
    """
    The benchmark benchmark evaluation.
    """

    def __init__(
        self,
        evaluator_l: List[FactorEvaluator],
        test_cases: List[TestCase],
        generate_method: Developer,
        catch_eval_except: bool = True,
    ):
        """Parameters
        ----------
        test_cases : List[TestCase]
            cases to be evaluated, ground truth are included in the test cases.
        evaluator_l : List[FactorEvaluator]
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
    ) -> List[Workspace]:
        path = Path(path)
        fi_l = []
        for tc in self.test_cases:
            try:
                fi = FactorFBWorkspace.from_folder(tc.task, path, **kwargs)
                fi_l.append(fi)
            except FileNotFoundError:
                print("Fail to load test case for factor: ", tc.task.factor_name)
        return fi_l

    def eval_case(
        self,
        case_gt: Workspace,
        case_gen: Workspace,
    ) -> List[Union[Tuple[FactorEvaluator, object], Exception]]:
        """Parameters
        ----------
        case_gt : FactorImplementation

        case_gen : FactorImplementation


        Returns
        -------
        List[Union[Tuple[FactorEvaluator, object],Exception]]
            for each item
                If the evaluation run successfully, return the evaluate results.  Otherwise, return the exception.
        """
        eval_res = []
        for ev in self.evaluator_l:
            try:
                eval_res.append((ev, ev.evaluate(implementation=case_gen, gt_implementation=case_gt)))
                # if the corr ev is successfully evaluated and achieve the best performance, then break
            except CoderError as e:
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
        test_cases: TestCase,
        method: Developer,
        *args,
        test_round: int = 10,
        **kwargs,
    ):
        online_evaluator_l = [
            FactorSingleColumnEvaluator(),
            FactorOutputFormatEvaluator(),
            FactorRowCountEvaluator(),
            FactorIndexEvaluator(),
            FactorMissingValuesEvaluator(),
            FactorEqualValueCountEvaluator(),
            FactorCorrelationEvaluator(hard_check=False),
        ]
        super().__init__(online_evaluator_l, test_cases, method, *args, **kwargs)
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
                gen_factor_l = self.generate_method.develop(self.test_cases.target_task)
            except KeyboardInterrupt:
                # TODO: Why still need to save result after KeyboardInterrupt?
                print("Manually interrupted the evaluation. Saving existing results")
                break

            if len(gen_factor_l.sub_workspace_list) != len(self.test_cases.ground_truth):
                raise ValueError(
                    "The number of cases to eval should be equal to the number of test cases.",
                )
            gen_factor_l_all_rounds.extend(gen_factor_l.sub_workspace_list)
            test_cases_all_rounds.extend(self.test_cases.ground_truth)

        eval_res_list = multiprocessing_wrapper(
            [
                (self.eval_case, (gt_case, gen_factor))
                for gt_case, gen_factor in zip(test_cases_all_rounds, gen_factor_l_all_rounds)
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )

        for gt_case, eval_res, gen_factor in tqdm(zip(test_cases_all_rounds, eval_res_list, gen_factor_l_all_rounds)):
            res[gt_case.target_task.factor_name].append((gen_factor, eval_res))

        return res
