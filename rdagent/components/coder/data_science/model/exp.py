from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask


# Because we use isinstance to distinguish between different types of tasks, we need to use sub classes to represent different types of tasks
class ModelTask(CoSTEERTask):
    def __init__(
        self,
        name: str,
        description: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name=name, description=description, *args, **kwargs)

    def get_task_information(self):
        task_desc = f"""name: {self.name}
description: {self.description}
"""
        return task_desc
