"""LLM Fine-tuning Base Classes"""

import json
from typing import Literal

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.components.coder.finetune.exp import TrainingTask
from rdagent.core.proposal import ExpGen, Hypothesis, Hypothesis2Experiment, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.experiment.experiment import FTExperiment
from rdagent.scenarios.finetune.scen.llama_factory_manager import (
    get_llama_factory_manager,
)
from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen
from rdagent.scenarios.finetune.scen.utils import extract_dataset_info
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T

COMPONENT = Literal["Training"]


class FTHypothesis(Hypothesis):
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


class FTHypothesis2Experiment(Hypothesis2Experiment):
    """Convert LLM fine-tuning hypothesis to experiment."""

    def convert(self, hypothesis: FTHypothesis, trace: Trace) -> FTExperiment:
        """Convert hypothesis to executable experiment."""
        logger.info(f"Converting hypothesis: {hypothesis.base_model} with {hypothesis.finetune_method}")

        # Download model at the point of use (after LLM selection)
        from rdagent.scenarios.finetune.utils import ensure_ft_assets_exist

        ensure_ft_assets_exist(model=hypothesis.base_model, check_model=True)

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

        # TODO: Single-stage task creation (future: multi-stage support)
        # Currently: [[task]] - one stage with one task
        # Future examples:
        #   - [[data_task], [train_task]] for data processing + training
        #   - [[sft_task], [rlhf_task]] for SFT + RLHF pipeline
        return FTExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)


class LLMFinetuneExpGen(ExpGen):
    """LLM fine-tuning experiment generator."""

    def __init__(self, scen: LLMFinetuneScen):
        super().__init__(scen)
        self.llama_manager = get_llama_factory_manager()

    def gen(self, trace: Trace, plan=None) -> FTExperiment:
        """Generate LLM fine-tuning experiment using LLM-driven selection."""
        device_info = get_runtime_environment_by_env(get_ft_env())
        dataset_info = extract_dataset_info(FT_RD_SETTING.dataset)

        device_dict = json.loads(device_info)
        memory_gb = device_dict.get("gpu", {}).get("total_gpu_memory_gb")
        logger.info(f"Device: {memory_gb}GB GPU")
        logger.info(f"Dataset: {dataset_info['name']}")

        # Check if model is specified in settings
        if FT_RD_SETTING.base_model:
            # Use specified model, but still select method and quantization intelligently
            logger.info(f"Using specified model: {FT_RD_SETTING.base_model}")
            base_model = FT_RD_SETTING.base_model
            _, finetune_method, quantization = self._llm_select_config(
                device_info, dataset_info, trace, specified_model=base_model
            )
        else:
            # Use LLM to intelligently select model, method and quantization
            logger.info("Auto-selecting optimal model configuration...")
            base_model, finetune_method, quantization = self._llm_select_config(device_info, dataset_info, trace)

        method_desc = finetune_method
        if quantization != "none":
            method_desc += f" with {quantization} quantization"

        hypothesis = FTHypothesis(
            component="Training",
            base_model=base_model,
            finetune_method=finetune_method,
            quantization=quantization,
            hypothesis=f"Fine-tune {base_model} using {method_desc} on {dataset_info['name']} dataset",
            reason=f"LLM-selected configuration for {memory_gb}GB GPU and dataset characteristics",
        )

        return FTHypothesis2Experiment().convert(hypothesis, trace)

    def _llm_select_config(
        self, device_info: str, dataset_info: dict, trace: Trace, specified_model: str | None = None
    ) -> tuple[str, str, str]:
        """Use LLM to intelligently select model, method and quantization.

        Args:
            device_info: Hardware information
            dataset_info: Dataset characteristics
            trace: Experiment trace
            specified_model: If provided, use this model and only select method/quantization

        Returns:
            Tuple of (model, method, quantization)
        """
        available_models = self.llama_manager.models
        available_methods = self.llama_manager.methods

        device_dict = json.loads(device_info)
        memory_gb = device_dict.get("gpu", {}).get("total_gpu_memory_gb")
        gpu_name = device_dict.get("gpu", {}).get("gpu_device", "Unknown")
        gpu_count = device_dict.get("gpu", {}).get("gpu_count", "Unknown")

        # Prepare template context
        template_context = {
            "memory_gb": memory_gb,
            "gpu_name": gpu_name,
            "gpu_count": gpu_count,
            "dataset_name": dataset_info["name"],
            "first_sample": dataset_info["samples"][0],
            "dataset_sample_count": dataset_info.get("sample_count", "Unknown"),
            "dataset_total_size_mb": dataset_info.get("total_size_mb", "Unknown"),
            "available_methods": ", ".join(available_methods),
            "specified_model": specified_model is not None,
        }

        if not specified_model:
            template_context["available_models"] = ", ".join(available_models[:10])  # Show first 10 models
            # TODO: show all models, maybe filtered by model size and gpu memory size
        else:
            template_context["specified_model"] = specified_model

        # Use template system for prompts
        # TODO: dicide more
        # TODO: separate first loop and other loop(need to get previous feedback)
        system_prompt = T(".prompts:hypothesis_gen.system_prompt").r(**template_context)
        user_prompt = T(".prompts:hypothesis_gen.user_prompt").r(**template_context)

        # Call LLM for selection
        from rdagent.oai.llm_utils import APIBackend

        llm = APIBackend()

        response = llm.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
            json_target_type=dict[str, str],
        )

        # Parse LLM response
        logger.info(f"LLM raw response: {response}")
        config = json.loads(response)

        selected_model = config["model"]
        selected_method = config["method"]
        selected_quantization = config["quantization"]

        # Validate selections
        if not specified_model and selected_model not in available_models:
            raise ValueError(
                f"LLM selected model '{selected_model}' is not in available models. "
                f"Available: {available_models[:5]}..."
            )
        elif specified_model:
            selected_model = specified_model  # Use specified model regardless of LLM output

        if selected_method not in available_methods:
            raise ValueError(
                f"LLM selected method '{selected_method}' is not in available methods. "
                f"Available: {available_methods}"
            )

        if selected_quantization not in ["none", "4bit", "8bit"]:
            raise ValueError(
                f"LLM selected quantization '{selected_quantization}' is not valid. " f"Valid options: none, 4bit, 8bit"
            )

        logger.info(f"LLM selected: {selected_model} + {selected_method} + {selected_quantization}")
        return selected_model, selected_method, selected_quantization
