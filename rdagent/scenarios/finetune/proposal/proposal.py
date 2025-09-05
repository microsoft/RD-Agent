"""
LLM Fine-tuning Base Classes

Contains core hypothesis and task classes for LLM fine-tuning scenarios.
"""

import json
import re
from typing import Literal

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.components.coder.finetune.exp import TrainingTask
from rdagent.core.experiment import Task
from rdagent.core.proposal import ExpGen, Hypothesis, Hypothesis2Experiment, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.experiment.experiment import FTExperiment
from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen
from rdagent.scenarios.finetune.scen.utils import extract_dataset_info
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env

# Available fine-tuning methods
AVAILABLE_FINETUNE_METHODS = [
    "full_params",
    "lora",
    "qlora",
    # TODO: Future methods can be added here:
    # "adapter", "ptuning_v2", "dora", "lora+", "galore", "longlora", "pissa", "rslora", "neftune"
]

# Available base models (simplified for debugging)
# TODO: Add more models here
AVAILABLE_BASE_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",  # Only one model for debugging phase
]

# Component definition for LLM fine-tuning (following data science pattern)
COMPONENT = Literal["Training"]


class LLMHypothesis(Hypothesis):
    """LLM fine-tuning hypothesis class - follows data science pattern with component"""

    def __init__(
        self,
        component: COMPONENT,
        base_model: str,
        finetune_method: str,
        hypothesis: str | None = None,
        reason: str | None = None,
        concise_reason: str | None = None,
        concise_observation: str | None = None,
        concise_justification: str | None = None,
        concise_knowledge: str | None = None,
    ) -> None:
        super().__init__(
            hypothesis, reason, concise_reason, concise_observation, concise_justification, concise_knowledge
        )
        self.component = component
        self.base_model = base_model
        self.finetune_method = finetune_method

    def __str__(self) -> str:
        if self.hypothesis is None:
            return f"No hypothesis available. Trying to construct the first runnable {self.component} component."

        lines = []
        lines.append(f"Component: {self.component}")
        lines.append(f"Base Model: {self.base_model}")
        lines.append(f"Fine-tuning Method: {self.finetune_method}")
        lines.append(f"Hypothesis: {self.hypothesis}")
        if self.reason is not None:
            lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


# TrainingTask is now imported from components.coder.finetune.exp
# This follows the data science pattern where proposal uses the same task classes as coder


class LLMHypothesis2Experiment(Hypothesis2Experiment):
    """Convert LLM fine-tuning hypothesis to experiment"""

    def convert(self, hypothesis: LLMHypothesis, trace: Trace) -> FTExperiment:
        """Convert hypothesis to executable experiment"""

        logger.info(
            f"Converting LLM hypothesis to experiment: {hypothesis.base_model} with {hypothesis.finetune_method}"
        )

        dataset = FT_RD_SETTING.dataset

        # Create fine-tuning task with only essential parameters
        task = TrainingTask(
            base_model=hypothesis.base_model,
            finetune_method=hypothesis.finetune_method,
            dataset=dataset,
            name="Training",
            description=f"Fine-tune {hypothesis.base_model} using {hypothesis.finetune_method} method",
        )

        # Create experiment
        experiment = FTExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)

        return experiment


class LLMFinetuneExpGen(ExpGen):
    """LLM fine-tuning experiment generator"""

    def __init__(self, scen: LLMFinetuneScen):
        super().__init__(scen)

    def gen(self, trace: Trace, plan=None) -> FTExperiment:
        """Generate LLM fine-tuning experiment"""

        # 1. Detect GPU capabilities and dataset information
        device_info = get_runtime_environment_by_env(get_ft_env())
        dataset_info = extract_dataset_info(FT_RD_SETTING.dataset)
        logger.info(f"Device detected: {device_info}")
        logger.info(f"Dataset: {dataset_info['name']}")

        # 2. Select appropriate base model and fine-tuning method based on both device and dataset
        base_model, finetune_method = self._select_model_and_method(device_info, dataset_info, trace)

        # 3. Create simple hypothesis (no hyperparameters)
        device_info_dict = json.loads(device_info)
        memory_gb = (
            device_info_dict["gpu"]["total_gpu_memory_gb"]
            if "gpu" in device_info_dict and "total_gpu_memory_gb" in device_info_dict["gpu"]
            else None
        )
        hypothesis = LLMHypothesis(
            component="Training",
            base_model=base_model,
            finetune_method=finetune_method,
            hypothesis=f"Fine-tune {base_model} using {finetune_method} method on {dataset_info['name']} dataset to improve capability",
            reason=f"Selected based on device capability ({memory_gb}GB GPU) and dataset characteristics",
        )

        # 4. Convert to experiment
        converter = LLMHypothesis2Experiment()
        experiment = converter.convert(hypothesis, trace)

        return experiment

    # TODO: decide by llm or hard code logic?
    # ATTENTION: This is a oversimplified version
    def _select_model_and_method(self, device_info: str, dataset_info: dict, trace: Trace) -> tuple[str, str]:
        """Select base model and fine-tuning method based on device capability and dataset"""
        device_info_dict = json.loads(device_info)
        memory_gb = (
            device_info_dict["gpu"]["total_gpu_memory_gb"]
            if "gpu" in device_info_dict and "total_gpu_memory_gb" in device_info_dict["gpu"]
            else None
        )

        # For debugging, only use the single available model
        base_model = AVAILABLE_BASE_MODELS[0]

        # Select fine-tuning method based on GPU memory
        if memory_gb is None:
            finetune_method = "qlora"
        elif memory_gb >= 12:
            finetune_method = "lora"  # More stable for debugging
        elif memory_gb >= 8:
            finetune_method = "qlora"  # More memory efficient
        else:
            finetune_method = "qlora"  # Fallback to most memory efficient

        logger.info(f"Selected {base_model} with {finetune_method} method for {memory_gb}GB GPU")
        return base_model, finetune_method
