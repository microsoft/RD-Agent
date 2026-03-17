"""RL Post-training Experiment"""

from rdagent.core.experiment import Experiment, Task
from rdagent.scenarios.rl.experiment.workspace import RLWorkspace


class RLTask(Task):
    """RDLoop 内部的任务描述（每次迭代一个）。

    仅用于 rdagent 框架内部流转，和 autorl_bench 的 benchmark 无关。
    """

    pass


class RLExperiment(Experiment[RLTask, RLWorkspace, RLWorkspace]):
    """RL post-training experiment with workspace initialization."""

    def __init__(self, sub_tasks: list[RLTask], *args, **kwargs) -> None:
        super().__init__(sub_tasks=sub_tasks, *args, **kwargs)
        # Initialize experiment workspace (required by CoSTEER)
        self.experiment_workspace = RLWorkspace()
