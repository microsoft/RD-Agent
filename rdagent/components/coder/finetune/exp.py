"""
LLM Fine-tuning Experiment Components

Defines tasks and experiments specific to LLM fine-tuning.
"""

from rdagent.components.coder.CoSTEER.task import CoSTEERTask


class LLMFinetuneTask(CoSTEERTask):
    """Task class for LLM fine-tuning operations"""

    def __init__(
        self,
        name: str = "LLMFinetune",
        base_model: str = "Qwen2.5-1.5B-Instruct",
        finetune_method: str = "lora",
        dataset: str = "default",
        description: str = "",
        debug_mode: bool = True,  # coding stage uses debug mode, running stage uses full mode
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name=name, description=description, *args, **kwargs)
        self.base_model = base_model
        self.finetune_method = finetune_method
        self.dataset = dataset
        self.debug_mode = debug_mode

    def get_task_information(self) -> str:
        """Get detailed task information for prompt generation"""
        mode = "Debug (100 samples)" if self.debug_mode else "Full training"
        info = f"""Task: {self.name}
Base Model: {self.base_model}
Fine-tuning Method: {self.finetune_method}
Dataset: {self.dataset}
Mode: {mode}
Description: {self.description}
"""
        return info.strip()
