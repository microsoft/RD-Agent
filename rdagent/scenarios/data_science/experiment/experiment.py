import re
from typing import Literal

import pandas as pd

from rdagent.core.experiment import Experiment, FBWorkspace, Task

COMPONENT = Literal["DataLoadSpec", "FeatureEng", "Model", "Ensemble", "Workflow", "Pipeline"]


class DSExperiment(Experiment[Task, FBWorkspace, FBWorkspace]):
    def __init__(self, pending_tasks_list: list, *args, **kwargs) -> None:
        super().__init__(sub_tasks=[], *args, **kwargs)
        # Status
        # - Initial: blank;
        # - Injecting from SOTA code;
        # - New version no matter successful or not
        # the initial workspace or the successful new version after coding
        self.experiment_workspace = FBWorkspace()
        self.pending_tasks_list = pending_tasks_list

        self.format_check_result = None
        # this field is optional. It  is not none only when we have a format checker. Currently, only following cases are supported.
        # - mle-bench

    def is_ready_to_run(self) -> bool:
        """
        ready to run does not indicate the experiment is runnable
        (so it is different from `trace.next_incomplete_component`.)
        """
        return self.experiment_workspace is not None and "main.py" in self.experiment_workspace.file_dict

    def set_local_selection(self, local_selection: tuple[int, ...]) -> None:
        self.local_selection = local_selection
