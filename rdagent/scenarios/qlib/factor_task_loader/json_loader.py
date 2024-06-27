import json
from pathlib import Path

from rdagent.components.task_implementation.factor_implementation.evolving.factor import (
    FactorImplementTask,
    FileBasedFactorImplementation,
)
from rdagent.components.task_loader import FactorTaskLoader
from rdagent.core.task import TaskLoader, TestCase


class FactorImplementationTaskLoaderFromDict(FactorTaskLoader):
    def load(self, factor_dict: dict) -> list:
        """Load data from a dict."""
        task_l = []
        for factor_name, factor_data in factor_dict.items():
            task = FactorImplementTask(
                factor_name=factor_name,
                factor_description=factor_data["description"],
                factor_formulation=factor_data["formulation"],
                variables=factor_data["variables"],
            )
            task_l.append(task)
        return task_l


class FactorImplementationTaskLoaderFromJsonFile(FactorTaskLoader):
    def load(self, json_file_path: Path) -> list:
        with open(json_file_path, "r") as file:
            factor_dict = json.load(file)
        return FactorImplementationTaskLoaderFromDict().load(factor_dict)


class FactorImplementationTaskLoaderFromJsonString(FactorTaskLoader):
    def load(self, json_string: str) -> list:
        factor_dict = json.loads(json_string)
        return FactorImplementationTaskLoaderFromDict().load(factor_dict)


class FactorTestCaseLoaderFromJsonFile(TaskLoader):
    def load(self, json_file_path: Path) -> list:
        with open(json_file_path, "r") as file:
            factor_dict = json.load(file)
        TestData = TestCase()
        for factor_name, factor_data in factor_dict.items():
            task = FactorImplementTask(
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
