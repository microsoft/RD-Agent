import pickle
import site
import traceback
from pathlib import Path
from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import DockerEnv, DSDockerConf


class DataLoaderTask(CoSTEERTask):
    def __init__(
        self,
        name: str,
        description: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name=name, description=description, *args, **kwargs)

    def get_task_information(self):
        task_desc = f"""name: {self.name}
description: {self.description}
"""
        return task_desc

    @staticmethod
    def from_dict(dict):
        return DataLoaderTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"
