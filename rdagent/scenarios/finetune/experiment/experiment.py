import re
from typing import Literal

import pandas as pd

from rdagent.core.experiment import Experiment, Task
from rdagent.scenarios.finetune.experiment.workspace import FTWorkspace

COMPONENT = Literal["Training"]


class FTExperiment(Experiment[Task, FTWorkspace, FTWorkspace]):
    def __init__(self, pending_tasks_list: list, *args, **kwargs) -> None:
        super().__init__(sub_tasks=[], *args, **kwargs)
        # Status
        # - Initial: blank;
        # - Injecting from SOTA code;
        # - New version no matter successful or not
        # the initial workspace or the successful new version after coding
        self.experiment_workspace = FTWorkspace()

        # TODO: Multi-stage task support (currently single-stage only)
        # Current: pending_tasks_list always contains exactly one task group [[TrainingTask]]
        # Future possibilities:
        #   - Data processing + Training: [[DataProcessTask], [TrainingTask]]
        #   - SFT + RLHF pipeline: [[SupervisedFinetuneTask], [RLHFTask]]
        #
        # When adding multi-stage support:
        #   1. Define stage dependencies in scenario or task classes
        #   2. Update loop.py coding() to iterate over stages like DataScience does
        #   3. Implement stage-specific coders if needed
        #
        # For now, this follows the same structure as DataScience for consistency,
        # even though FT only executes single-stage tasks per loop iteration.
        self.pending_tasks_list = pending_tasks_list

        self.format_check_result = None
        # this field is optional. It is not none only when we have a format checker. Currently, only following cases are supported.
        # - mle-bench

    def is_ready_to_run(self) -> bool:
        """
        ready to run does not indicate the experiment is runnable
        (so it is different from `trace.next_incomplete_component`.)
        """
        return self.experiment_workspace is not None and "main.py" in self.experiment_workspace.file_dict

    def set_local_selection(self, local_selection: tuple[int, ...]) -> None:
        self.local_selection = local_selection
