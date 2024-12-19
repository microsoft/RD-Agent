from rdagent.core.experiment import Experiment, FBWorkspace, Task

class DSExperiment(Experiment[Task, FBWorkspace, FBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = FBWorkspace()
