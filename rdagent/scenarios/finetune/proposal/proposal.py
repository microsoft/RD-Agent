"""
LLM Fine-tuning Base Classes

Contains core hypothesis and task classes for LLM fine-tuning scenarios.
"""

import re

from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.core.experiment import Task
from rdagent.core.proposal import ExpGen, Hypothesis, Hypothesis2Experiment, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen
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


class LLMHypothesis(Hypothesis):
    """LLM fine-tuning hypothesis class - only specifies model and method"""

    def __init__(
        self,
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
        self.base_model = base_model
        self.finetune_method = finetune_method

    def __str__(self) -> str:
        if self.hypothesis is None:
            return f"Fine-tune {self.base_model} model using {self.finetune_method} method"

        lines = []
        lines.append(f"Base Model: {self.base_model}")
        lines.append(f"Fine-tuning Method: {self.finetune_method}")
        lines.append(f"Hypothesis: {self.hypothesis}")
        if self.reason is not None:
            lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


class LLMFinetuneTask(Task):
    """LLM fine-tuning task class"""

    def __init__(
        self,
        base_model: str,
        finetune_method: str,
        dataset: str = "default",
        name: str = "LLMFineTune",
        description: str = "",
        **kwargs,
    ):
        super().__init__(name=name, description=description, **kwargs)
        self.base_model = base_model
        self.finetune_method = finetune_method
        self.dataset = dataset


class LLMHypothesis2Experiment(Hypothesis2Experiment):
    """Convert LLM fine-tuning hypothesis to experiment"""

    def convert(self, hypothesis: LLMHypothesis, trace: Trace) -> DSExperiment:
        """Convert hypothesis to executable experiment"""

        logger.info(
            f"Converting LLM hypothesis to experiment: {hypothesis.base_model} with {hypothesis.finetune_method}"
        )

        # Get dataset from settings
        try:
            from rdagent.app.finetune.llm.conf import FT_RD_SETTING

            dataset = FT_RD_SETTING.dataset if hasattr(FT_RD_SETTING, "dataset") else "default"
        except:
            dataset = "default"

        # Create fine-tuning task with only essential parameters
        task = LLMFinetuneTask(
            base_model=hypothesis.base_model,
            finetune_method=hypothesis.finetune_method,
            dataset=dataset,
            name="LLMFineTune",
            description=f"Fine-tune {hypothesis.base_model} using {hypothesis.finetune_method} method",
        )

        # Create experiment
        experiment = DSExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)

        return experiment


class LLMFinetuneExpGen(ExpGen):
    """LLM fine-tuning experiment generator"""

    def __init__(self, scen: LLMFinetuneScen):
        super().__init__(scen)

    def gen(self, trace: Trace, plan=None) -> DSExperiment:
        """Generate LLM fine-tuning experiment"""

        # 1. Detect GPU capabilities and dataset information
        device_info = get_runtime_environment_by_env(get_ft_env())
        dataset_info = self._get_dataset_info()
        logger.info(f"Device detected: {device_info}")
        logger.info(f"Dataset: {dataset_info['name']}")

        # 2. Select appropriate base model and fine-tuning method based on both device and dataset
        base_model, finetune_method = self._select_model_and_method(device_info, dataset_info, trace)

        # 3. Create simple hypothesis (no hyperparameters)
        hypothesis = LLMHypothesis(
            base_model=base_model,
            finetune_method=finetune_method,
            hypothesis=f"Fine-tune {base_model} using {finetune_method} method on {dataset_info['name']} dataset to improve capability",
            reason=f"Selected based on device capability ({device_info.get('memory_gb', 'unknown')}GB GPU) and dataset characteristics",
        )

        # 4. Convert to experiment
        converter = LLMHypothesis2Experiment()
        experiment = converter.convert(hypothesis, trace)

        return experiment

    def _get_dataset_info(self) -> dict:
        """Get dataset information using existing utility"""
        try:
            from rdagent.app.finetune.llm.conf import FT_RD_SETTING
            from rdagent.scenarios.finetune.scen.utils import extract_dataset_info

            dataset_name = FT_RD_SETTING.dataset if hasattr(FT_RD_SETTING, "dataset") else "default"
            return extract_dataset_info(dataset_name)
        except Exception as e:
            logger.warning(f"Failed to get dataset info: {e}")
            return {"name": "unknown", "description": "", "samples": [], "files": []}

    # TODO: decide by llm or hard code logic?
    # ATTENTION: This is a oversimplified version
    def _select_model_and_method(self, device_info: str, dataset_info: dict, trace: Trace) -> tuple[str, str]:
        """Select base model and fine-tuning method based on device capability and dataset"""
        memory_gb = re.search(r"Total GPU Memory:\s*([\d.]+)\s*GB", device_info)
        memory_gb = float(memory_gb.group(1)) if memory_gb else None

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
