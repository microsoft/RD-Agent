import re
from typing import Literal

import pandas as pd

from rdagent.core.experiment import Experiment, FBWorkspace, Task

COMPONENT = Literal["DataLoadSpec", "FeatureEng", "Model", "Ensemble", "Workflow"]


class DSExperiment(Experiment[Task, FBWorkspace, FBWorkspace]):
    def __init__(self, pending_tasks_list: list, *args, **kwargs) -> None:
        super().__init__(sub_tasks=[], *args, **kwargs)
        self.experiment_workspace = FBWorkspace()
        self.pending_tasks_list = pending_tasks_list

    def next_component_required(self) -> COMPONENT | None:
        files = list(self.experiment_workspace.file_dict.keys())
        if "load_data.py" not in files:
            return "DataLoadSpec"
        if "feature.py" not in files:
            return "FeatureEng"
        if not any(re.match(r"model.*\.py", file) for file in files):
            return "Model"
        if "ensemble.py" not in files:
            return "Ensemble"
        if "main.py" not in files:
            return "Workflow"
        return None
