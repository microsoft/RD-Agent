import pickle
import site
import traceback
from pathlib import Path
from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.core.utils import cache_with_pickle


class WorkflowTask(CoSTEERTask):
    def __init__(
        self,
        name: str,
        description: str,
        **kwargs,
    ) -> None:
        pass

    @staticmethod
    def from_dict(dict):
        return WorkflowTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"
