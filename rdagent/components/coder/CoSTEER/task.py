from rdagent.core.experiment import Task


class CoSTEERTask(Task):
    def __init__(self, base_code: str = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: we may upgrade the base_code into a workspace-like thing to know previous.
        # NOTE: (xiao) think we don't need the base_code anymore. The information should be retrieved from the workspace.
        self.base_code = base_code
