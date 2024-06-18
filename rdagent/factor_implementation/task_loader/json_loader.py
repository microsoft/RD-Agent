import json
from pathlib import Path
from rdagent.core.task import TaskLoader
from rdagent.factor_implementation.evolving.factor import FactorImplementTask


class FactorImplementationTaskLoaderFromDict(TaskLoader):
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


class FactorImplementationTaskLoaderFromJsonFile(TaskLoader):
    def load(self, json_file_path: Path) -> list:
        factor_dict = json.load(json_file_path)
        return FactorImplementationTaskLoaderFromDict().load(factor_dict)


class FactorImplementationTaskLoaderFromJsonString(TaskLoader):
    def load(self, json_string: str) -> list:
        factor_dict = json.loads(json_string)
        return FactorImplementationTaskLoaderFromDict().load(factor_dict)
