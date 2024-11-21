from rdagent.core.experiment import Task


class CoSTEERTask(Task):
    def __init__(self, base_code: str = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_code = base_code
