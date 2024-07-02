import json
from pathlib import Path

from rdagent.components.benchmark.eval_method import TestCase
from rdagent.components.loader.experiment_loader import FactorExperimentLoader
from rdagent.components.task_implementation.factor_implementation.factor import (
    FactorExperiment,
    FactorTask,
    FileBasedFactorImplementation,
)
from rdagent.core.experiment import Loader


class FactorImplementationExperimentLoaderFromDict(FactorExperimentLoader):
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
        exp = FactorExperiment(sub_tasks=task_l)
        return exp


class FactorImplementationExperimentLoaderFromJsonFile(FactorExperimentLoader):
    def load(self, json_file_path: Path) -> list:
        with open(json_file_path, "r") as file:
            factor_dict = json.load(file)
        return FactorImplementationExperimentLoaderFromDict().load(factor_dict)


class FactorImplementationExperimentLoaderFromJsonString(FactorExperimentLoader):
    def load(self, json_string: str) -> list:
        factor_dict = json.loads(json_string)
        return FactorImplementationExperimentLoaderFromDict().load(factor_dict)


# TODO loader only supports generic of task or experiment, testcase might cause CI error here
# class FactorTestCaseLoaderFromJsonFile(Loader[TestCase]):
class FactorTestCaseLoaderFromJsonFile:
    def load(self, json_file_path: Path) -> list:
        with open(json_file_path, "r") as file:
            factor_dict = json.load(file)
        TestData = TestCase()
        for factor_name, factor_data in factor_dict.items():
            task = FactorTask(
                factor_name=factor_name,
                factor_description=factor_data["description"],
                factor_formulation=factor_data["formulation"],
                variables=factor_data["variables"],
            )
            gt = FileBasedFactorImplementation(task, code=factor_data["gt_code"])
            gt.execute()
            TestData.target_task.append(task)
            TestData.ground_truth.append(gt)

        return TestData
