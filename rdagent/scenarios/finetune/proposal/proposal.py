"""LLM Fine-tuning Base Classes"""

import json
from typing import Literal

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.components.coder.finetune.exp import TrainingTask
from rdagent.core.proposal import ExpGen, Hypothesis, Hypothesis2Experiment, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.experiment.experiment import FTExperiment
from rdagent.scenarios.finetune.llama_factory_manager import LLaMAFactoryManager
from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen
from rdagent.scenarios.finetune.scen.utils import extract_dataset_info
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env

COMPONENT = Literal["Training"]


class LLMHypothesis(Hypothesis):
    """LLM fine-tuning hypothesis class."""

    def __init__(
        self,
        component: COMPONENT,
        base_model: str,
        finetune_method: str,
        quantization: str = "none",
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
        self.quantization = quantization

    def __str__(self) -> str:
        if self.hypothesis is None:
            return f"No hypothesis available. Constructing first runnable {self.component} component."

        lines = [
            f"Component: {self.component}",
            f"Base Model: {self.base_model}",
            f"Fine-tuning Method: {self.finetune_method}",
        ]
        if self.quantization != "none":
            lines.append(f"Quantization: {self.quantization}")
        lines.append(f"Hypothesis: {self.hypothesis}")
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


class LLMHypothesis2Experiment(Hypothesis2Experiment):
    """Convert LLM fine-tuning hypothesis to experiment."""

    def convert(self, hypothesis: LLMHypothesis, trace: Trace) -> FTExperiment:
        """Convert hypothesis to executable experiment."""
        logger.info(f"Converting hypothesis: {hypothesis.base_model} with {hypothesis.finetune_method}")

        # Combine method and quantization for task description
        method_desc = hypothesis.finetune_method
        if hypothesis.quantization != "none":
            method_desc += f" with {hypothesis.quantization} quantization"

        task = TrainingTask(
            base_model=hypothesis.base_model,
            finetune_method=hypothesis.finetune_method,
            dataset=FT_RD_SETTING.dataset,
            name="Training",
            description=f"Fine-tune {hypothesis.base_model} using {method_desc}",
        )

        return FTExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)


class LLMFinetuneExpGen(ExpGen):
    """LLM fine-tuning experiment generator."""

    def __init__(self, scen: LLMFinetuneScen):
        super().__init__(scen)
        self.llama_manager = LLaMAFactoryManager()

    def gen(self, trace: Trace, plan=None) -> FTExperiment:
        """Generate LLM fine-tuning experiment using LLM-driven selection."""
        device_info = get_runtime_environment_by_env(get_ft_env())
        dataset_info = extract_dataset_info(FT_RD_SETTING.dataset)

        device_dict = json.loads(device_info)
        memory_gb = device_dict.get("gpu", {}).get("total_gpu_memory_gb")
        logger.info(f"Device: {memory_gb}GB GPU")
        logger.info(f"Dataset: {dataset_info['name']}")

        # Use LLM to intelligently select model, method and quantization
        base_model, finetune_method, quantization = self._llm_select_config(device_info, dataset_info, trace)

        method_desc = finetune_method
        if quantization != "none":
            method_desc += f" with {quantization} quantization"

        hypothesis = LLMHypothesis(
            component="Training",
            base_model=base_model,
            finetune_method=finetune_method,
            quantization=quantization,
            hypothesis=f"Fine-tune {base_model} using {method_desc} on {dataset_info['name']} dataset",
            reason=f"LLM-selected configuration for {memory_gb}GB GPU and dataset characteristics",
        )

        return LLMHypothesis2Experiment().convert(hypothesis, trace)

    def _llm_select_config(self, device_info: str, dataset_info: dict, trace: Trace) -> tuple[str, str, str]:
        """Use LLM to intelligently select model, method and quantization."""
        available_models = self.llama_manager.models
        available_methods = self.llama_manager.methods

        device_dict = json.loads(device_info)
        memory_gb = device_dict.get("gpu", {}).get("total_gpu_memory_gb")
        gpu_name = device_dict.get("gpu", {}).get("name", "Unknown")

        # Prepare context for LLM
        context = f"""Task: Select optimal configuration for LLM fine-tuning

GPU Information:
- Memory: {memory_gb}GB
- Name: {gpu_name}

Dataset Information:
- Name: {dataset_info['name']}
- Type: {dataset_info.get('type', 'Unknown')}
- Size: {dataset_info.get('size', 'Unknown')} samples

Available Models:
{', '.join(available_models[:10])}  # Show first 10 models

Available Fine-tuning Methods:
{', '.join(available_methods)}

Quantization Options:
- none: No quantization (highest quality, most memory)
- 4bit: 4-bit quantization (good balance)
- 8bit: 8-bit quantization (memory efficient)

Please select the optimal combination considering GPU memory constraints and task requirements.
Format your response as: model_name|method|quantization

Example: Qwen/Qwen2.5-1.5B-Instruct|lora|4bit"""

        try:
            # Call LLM for selection
            from rdagent.oai.llm_utils import APIBackend

            llm = APIBackend()

            response = llm.build_messages_and_create_chat_completion(
                user_prompt=context,
                system_prompt="You are an expert in LLM fine-tuning. Select the optimal configuration based on hardware constraints and dataset characteristics.",
            )

            # Parse LLM response
            response_text = response.strip()
            if "|" in response_text:
                parts = response_text.split("|")
                if len(parts) >= 3:
                    selected_model = parts[0].strip()
                    selected_method = parts[1].strip()
                    selected_quantization = parts[2].strip()

                    # Validate selections
                    if selected_model not in available_models:
                        selected_model = available_models[0]
                    if selected_method not in available_methods:
                        selected_method = "lora" if "lora" in available_methods else available_methods[0]
                    if selected_quantization not in ["none", "4bit", "8bit"]:
                        selected_quantization = "none"

                    logger.info(f"LLM selected: {selected_model} + {selected_method} + {selected_quantization}")
                    return selected_model, selected_method, selected_quantization
        except Exception as e:
            logger.warning(f"LLM selection failed, using fallback: {e}")

        # Fallback selection logic
        base_model = available_models[0] if available_models else "Qwen/Qwen2.5-1.5B-Instruct"
        finetune_method = "lora" if "lora" in available_methods else available_methods[0]
        quantization = "4bit" if memory_gb and memory_gb < 16 else "none"

        logger.info(f"Fallback selected: {base_model} + {finetune_method} + {quantization}")
        return base_model, finetune_method, quantization
