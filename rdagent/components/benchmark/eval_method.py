from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Union

from tqdm import tqdm

from rdagent.components.task_implementation.factor_implementation.evolving.evaluators import (
    FactorImplementationCorrelationEvaluator,
    FactorImplementationEvaluator,
    FactorImplementationIndexEvaluator,
    FactorImplementationIndexFormatEvaluator,
    FactorImplementationMissingValuesEvaluator,
    FactorImplementationRowCountEvaluator,
    FactorImplementationSingleColumnEvaluator,
    FactorImplementationValuesEvaluator,
)
from rdagent.components.task_implementation.factor_implementation.evolving.factor import (
    FileBasedFactorImplementation,
)
from rdagent.components.task_implementation.factor_implementation.share_modules.factor_implementation_config import (
    FACTOR_IMPLEMENT_SETTINGS,
)
from rdagent.core.exception import ImplementRunException
from rdagent.core.implementation import TaskGenerator
from rdagent.core.task import TaskImplementation, TestCase
from rdagent.core.utils import multiprocessing_wrapper


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
        test_cases: TestCase,
        method: TaskGenerator,
        test_round: int = 10,
        *args,
        **kwargs,
    ):
        online_evaluator_l = [
            FactorImplementationSingleColumnEvaluator(),
            FactorImplementationIndexFormatEvaluator(),
            FactorImplementationRowCountEvaluator(),
            FactorImplementationIndexEvaluator(),
            FactorImplementationMissingValuesEvaluator(),
            FactorImplementationValuesEvaluator(),
            FactorImplementationCorrelationEvaluator(hard_check=False),
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
                gen_factor_l = self.generate_method.generate(self.test_cases.target_task)
            except KeyboardInterrupt:
                # TODO: Why still need to save result after KeyboardInterrupt?
                print("Manually interrupted the evaluation. Saving existing results")
                break

            if len(gen_factor_l.corresponding_implementations) != len(self.test_cases.ground_truth):
                raise ValueError(
                    "The number of cases to eval should be equal to the number of test cases.",
                )
            gen_factor_l_all_rounds.extend(gen_factor_l.corresponding_implementations)
            test_cases_all_rounds.extend(self.test_cases.ground_truth)

        eval_res_list = multiprocessing_wrapper(
            [
                (self.eval_case, (gt_case, gen_factor))
                for gt_case, gen_factor in zip(test_cases_all_rounds, gen_factor_l_all_rounds)
            ],
            n=FACTOR_IMPLEMENT_SETTINGS.evo_multi_proc_n,
        )

        for gt_case, eval_res, gen_factor in tqdm(zip(test_cases_all_rounds, eval_res_list, gen_factor_l_all_rounds)):
            res[gt_case.target_task.factor_name].append((gen_factor, eval_res))

        return res
