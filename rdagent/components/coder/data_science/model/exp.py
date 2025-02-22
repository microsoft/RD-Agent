import pickle
import site
import traceback
from pathlib import Path
from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils.env import DockerEnv, DSDockerConf


# Because we use isinstance to distinguish between different types of tasks, we need to use sub classes to represent different types of tasks
class ModelTask(CoSTEERTask):
    def __init__(
        self,
        name: str,
        description: str,
        architecture: str = "",
        *args,
        hyperparameters: Dict[str, str] = {},
        model_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.architecture: str = architecture
        self.hyperparameters: str = hyperparameters
        self.model_type: str | None = (
            model_type  # Tabular for tabular model, TimesSeries for time series model, Graph for graph model, XGBoost for XGBoost model
            # TODO: More Models Supported
        )
        super().__init__(name=name, description=description, *args, **kwargs)

    def get_task_information(self):
        task_desc = f"""name: {self.name}
description: {self.description}
"""
        if self.architecture:
            task_desc += f"architecture: {self.architecture}\n"
        if self.hyperparameters:
            task_desc += f"hyperparameters: {self.hyperparameters}\n"
        if self.model_type:
            task_desc += f"model_type: {self.model_type}\n"
        return task_desc
