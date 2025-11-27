"""
LLM Fine-tuning Experiment Components

Defines tasks for LLM fine-tuning following data science pattern.
"""

from typing import List, Literal, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask


# Because we use isinstance to distinguish between different types of tasks, we need to use sub classes to represent different types of tasks
class FTTask(CoSTEERTask):
    """Training task class for LLM fine-tuning operations - follows data science pattern"""

    def __init__(
        self,
        base_model: str,
        description: str,
        benchmark: str,
        task_type: Literal["train", "data", "both"] = "train",
        involving_datasets: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name="LLM-Fine-Tuning", description=description, *args, **kwargs)
        self.base_model = base_model
        self.benchmark = benchmark
        self.task_type = task_type
        self.involving_datasets = involving_datasets or []

    def get_task_information(self) -> str:
        """Get task information for coder prompt generation"""
        task_desc = f"""name: {self.name}
description: {self.description}
base_model: {self.base_model}
task_type: {self.task_type}
"""
        # Add involving_datasets info for data/both tasks
        if self.task_type in ["data", "both"] and self.involving_datasets:
            task_desc += f"involving_datasets: {self.involving_datasets}\n"
        return task_desc
