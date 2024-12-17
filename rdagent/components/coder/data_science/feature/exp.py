import pickle
import site
import traceback
from pathlib import Path
from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.core.utils import cache_with_pickle


class FeatureTask(CoSTEERTask):
    def __init__(
        self,
        name: str,
        description: str,
        variables: dict = {},
        implementation: bool = False,
        **kwargs,
    ) -> None:
        self.variables: dict = variables
        self.implementation: bool = implementation
        super().__init__(name=name, description=description, **kwargs)

    def get_task_information(self):
        return f"""name: {self.name}
description: {self.description}
variables: {str(self.variables)}
spec: {self.spec}"""

    def get_task_information_and_implementation_result(self):
        return {
            "name": self.factor_name,
            "description": self.factor_description,
            "variables": str(self.variables),
            "spec": self.spec,
            "implementation": str(self.implementation),
        }

    @staticmethod
    def from_dict(dict):
        return FeatureTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"
