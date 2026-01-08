"""
LLM Fine-tuning Experiment Components

Defines tasks for LLM fine-tuning following data science pattern.
"""

from typing import List, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask


class FTTask(CoSTEERTask):
    """Training task class for LLM fine-tuning operations - follows data science pattern"""

    def __init__(
        self,
        base_model: str,
        description: str,
        benchmark: str,
        involving_datasets: Optional[List[str]] = None,
        skip_data_processing: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name="LLM-Fine-Tuning", description=description, *args, **kwargs)
        self.base_model = base_model
        self.benchmark = benchmark
        self.involving_datasets = involving_datasets or []
        self.skip_data_processing = skip_data_processing  # If True, reuse SOTA's data processing script

    def get_task_information(self) -> str:
        """Get task information for coder prompt generation"""
        task_desc = f"""name: {self.name}
description: {self.description}
base_model: {self.base_model}
"""
        if self.involving_datasets:
            task_desc += f"involving_datasets: {self.involving_datasets}\n"
        return task_desc
