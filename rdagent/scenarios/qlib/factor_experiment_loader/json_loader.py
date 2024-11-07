import json
from pathlib import Path

from rdagent.components.benchmark.eval_method import TestCase, TestCases
from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
from rdagent.components.loader.experiment_loader import FactorExperimentLoader
from rdagent.core.experiment import Experiment, Loader
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment


class FactorExperimentLoaderFromDict(FactorExperimentLoader):
    def load(self, factor_dict: dict) -> list:
        """Load data from a dict."""
        task_l = []
        for factor_name, factor_data in factor_dict.items():
            task = FactorTask(
                factor_name=factor_name,
                factor_description=factor_data["description"],
                factor_formulation=factor_data["formulation"],
                variables=factor_data["variables"],
            )
            task_l.append(task)
        exp = QlibFactorExperiment(sub_tasks=task_l)
        return exp


class FactorExperimentLoaderFromJsonFile(FactorExperimentLoader):
    def load(self, json_file_path: Path) -> list:
        with open(json_file_path, "r") as file:
            factor_dict = json.load(file)
        return FactorExperimentLoaderFromDict().load(factor_dict)


class FactorExperimentLoaderFromJsonString(FactorExperimentLoader):
    def load(self, json_string: str) -> list:
        factor_dict = json.loads(json_string)
        return FactorExperimentLoaderFromDict().load(factor_dict)


# TODO loader only supports generic of task or experiment, testcase might cause CI error here
# class FactorTestCaseLoaderFromJsonFile(Loader[TestCases]):
class FactorTestCaseLoaderFromJsonFile:
    def load(self, json_file_path: Path) -> TestCases:
        with open(json_file_path, "r") as file:
            factor_dict = json.load(file)
        test_cases = TestCases()
        for factor_name, factor_data in factor_dict.items():
            task = FactorTask(
                factor_name=factor_name,
                factor_description=factor_data["description"],
                factor_formulation=factor_data["formulation"],
                variables=factor_data["variables"],
            )
            gt = FactorFBWorkspace(task, raise_exception=False)
            code = {"factor.py": factor_data["gt_code"]}
            gt.inject_code(**code)
            test_cases.test_case_l.append(TestCase(task, gt))

        return test_cases
