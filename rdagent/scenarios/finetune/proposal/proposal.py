"""LLM Fine-tuning Base Classes"""

import json
from typing import Literal

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.components.coder.finetune.exp import LLMFTTask
from rdagent.core.proposal import ExpGen, Hypothesis, Hypothesis2Experiment, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.experiment.experiment import FTExperiment
from rdagent.scenarios.finetune.scen.llama_factory_manager import (
    get_llama_factory_manager,
)
from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen
from rdagent.scenarios.finetune.scen.utils import extract_dataset_info
from rdagent.scenarios.finetune.utils import ensure_ft_assets_exist
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T

COMPONENT = Literal["Training"]


class FTHypothesis(Hypothesis):
    """LLM fine-tuning hypothesis class."""

    def __init__(
        self,
        component: COMPONENT,
        base_model: str,
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

    def __str__(self) -> str:
        if self.hypothesis is None:
            return f"No hypothesis available. Constructing first runnable {self.component} component."

        lines = [
            f"Component: {self.component}",
            f"Base Model: {self.base_model}",
            f"Hypothesis: {self.hypothesis}",
        ]
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


class FTHypothesis2Experiment(Hypothesis2Experiment):
    """Convert LLM fine-tuning hypothesis to experiment."""

    def convert(self, hypothesis: FTHypothesis, trace: Trace) -> FTExperiment:
        """Convert hypothesis to executable experiment."""
        logger.info(f"Converting hypothesis for model: {hypothesis.base_model}")

        ensure_ft_assets_exist(model=hypothesis.base_model, check_model=True)

        task = LLMFTTask(
            base_model=hypothesis.base_model,
            dataset=FT_RD_SETTING.dataset,
            hypothesis=hypothesis.hypothesis,
            name="Training",
            description=f"Fine-tune {hypothesis.base_model} based on hypothesis",
        )

        return FTExperiment(sub_tasks=[task], hypothesis=hypothesis)


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

        # Generate hypothesis using LLM (model is always specified in current use case)
        logger.info(f"Using specified model: {FT_RD_SETTING.base_model}")
        base_model = FT_RD_SETTING.base_model
        base_model, hypothesis_text = self._llm_select_config(
            device_info, dataset_info, trace, specified_model=base_model
        )

        # Create hypothesis object with natural language description
        hypothesis = FTHypothesis(
            component="Training",
            base_model=base_model,
            hypothesis=hypothesis_text,
            reason=f"LLM-generated hypothesis for {memory_gb}GB GPU and {dataset_info['name']} dataset",
        )

        return FTHypothesis2Experiment().convert(hypothesis, trace)

    def _llm_select_config(
        self,
        device_info: str,
        dataset_info: dict,
        trace: Trace,
        specified_model: str | None = None,
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

        # Generate parameter descriptions: shared params once + method-specific params
        shared_params = self.llama_manager.format_shared_params()
        methods_specific_params = {}
        for method in available_methods:
            methods_specific_params[method] = self.llama_manager.format_method_specific_params(method)

        # Prepare template context
        template_context = {
            "memory_gb": memory_gb,
            "gpu_name": gpu_name,
            "gpu_count": gpu_count,
            "dataset_name": dataset_info["name"],
            "dataset_sample_count": dataset_info.get("sample_count", "Unknown"),
            "dataset_total_size_mb": dataset_info.get("total_size_mb", "Unknown"),
            "available_methods": ", ".join(available_methods),
            "shared_params": shared_params,
            "methods_specific_params": methods_specific_params,
            "specified_model": specified_model is not None,
            "trace": trace,  # Pass trace object directly
        }

        if not specified_model:
            template_context["available_models"] = ", ".join(available_models[:10])  # Show first 10 models
        else:
            template_context["specified_model"] = specified_model

        # Use template system for prompts
        system_prompt = T(".prompts:hypothesis_gen.system_prompt").r(**template_context)
        user_prompt = T(".prompts:hypothesis_gen.user_prompt").r(**template_context)

        # Call LLM to generate natural language hypothesis
        from rdagent.oai.llm_utils import APIBackend

        llm = APIBackend()

        hypothesis_text = llm.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
        )

        logger.info(f"LLM generated hypothesis:\n{hypothesis_text}")

        # Return model and hypothesis text
        # Method and quantization will be decided by Coder based on the hypothesis
        selected_model = specified_model
        return selected_model, hypothesis_text
