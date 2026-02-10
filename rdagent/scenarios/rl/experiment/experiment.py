"""RL Post-training Experiment"""

from rdagent.core.experiment import Experiment, Task
from rdagent.scenarios.rl.experiment.workspace import RLWorkspace

# TODO: maybe we can ke the class simple;
# if we need functions like `is_ready_to_run`, we create a standalone function. e.g.:
# def is_ready_to_run(exp) -> bool:
#     """Check if experiment is ready to run."""
#     return exp.experiment_workspace is not None and "main.py" in self.experiment_workspace.file_dict

class RLExperiment(Experiment[Task, RLWorkspace, RLWorkspace]):
    """RL post-training experiment with workspace initialization."""

    def __init__(self, sub_tasks: list[Task], *args, **kwargs) -> None:
        super().__init__(sub_tasks=sub_tasks, *args, **kwargs)
        # Initialize experiment workspace (required by CoSTEER)
        self.experiment_workspace = RLWorkspace()
