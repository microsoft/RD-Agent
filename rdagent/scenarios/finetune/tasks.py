"""
Simplified LLM fine-tuning task definitions
"""

from dataclasses import dataclass
from typing import Any, Dict

from rdagent.components.coder.data_science.pipeline.exp import PipelineTask


@dataclass
class DataFormatTask(PipelineTask):
    """Data format conversion task: convert raw dataset to LLaMA-Factory format"""

    name: str = "DataFormat"
    description: str = "Convert raw dataset to LLaMA-Factory format"
    dataset_name: str = ""

    def get_task_information(self) -> str:
        return f"DataFormat_{self.dataset_name}"


@dataclass
class FineTuningTask(PipelineTask):
    """Fine-tuning task: use LLaMA-Factory for model fine-tuning"""

    name: str = "FineTuning"
    description: str = "Fine-tune model using LLaMA-Factory"
    model_name: str = ""
    dataset_name: str = ""

    def get_task_information(self) -> str:
        return f"FineTuning_{self.model_name}_{self.dataset_name}"


def create_llm_finetune_tasks(dataset: str, model: str) -> list[PipelineTask]:
    """Create LLM fine-tuning task list"""

    data_task = DataFormatTask(
        name="DataFormat",
        description=f"Convert dataset {dataset} to LLaMA-Factory format",
        dataset_name=dataset,
    )

    finetune_task = FineTuningTask(
        name="FineTuning",
        description=f"Fine-tune {model} on processed {dataset}",
        model_name=model,
        dataset_name=dataset,
    )

    return [data_task, finetune_task]
