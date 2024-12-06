from rdagent.core.experiment import Task


class CoSTEERTask(Task):
    def __init__(self, base_code: str = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: we may upgrade the base_code into a workspace-like thing to know previous.
        self.base_code = base_code
