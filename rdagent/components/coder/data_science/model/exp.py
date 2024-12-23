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


class ModelTask(CoSTEERTask):
    def __init__(
        self,
        name: str,
        description: str,
        architecture: str,
        *args,
        hyperparameters: Dict[str, str],
        formulation: str = None,
        variables: Dict[str, str] = None,
        model_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.formulation: str = formulation
        self.architecture: str = architecture
        self.variables: str = variables
        self.hyperparameters: str = hyperparameters
        self.model_type: str = (
            model_type  # Tabular for tabular model, TimesSeries for time series model, Graph for graph model, XGBoost for XGBoost model
            # TODO: More Models Supported
        )
        super().__init__(name=name, description=description, *args, **kwargs)

    def get_task_information(self):
        task_desc = f"""name: {self.name}
description: {self.description}
"""
        task_desc += f"formulation: {self.formulation}\n" if self.formulation else ""
        task_desc += f"architecture: {self.architecture}\n"
        task_desc += f"variables: {self.variables}\n" if self.variables else ""
        task_desc += f"hyperparameters: {self.hyperparameters}\n"
        task_desc += f"model_type: {self.model_type}\n"
        return task_desc
