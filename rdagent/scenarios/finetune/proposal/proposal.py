"""LLM Fine-tuning Base Classes"""

import json

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.exp import FTTask
from rdagent.core.proposal import ExpGen, Hypothesis, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.experiment.experiment import FTExperiment
from rdagent.scenarios.finetune.scen.llama_factory_manager import (
    get_llama_factory_manager,
)
from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen
from rdagent.scenarios.finetune.utils import ensure_ft_assets_exist
from rdagent.utils.agent.tpl import T
from rdagent.oai.llm_utils import APIBackend


class FTHypothesis(Hypothesis):
    """LLM fine-tuning hypothesis class."""

    def __init__(
        self,
        base_model: str,
        hypothesis: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(
            hypothesis,
            reason,
            concise_reason="",
            concise_observation="",
            concise_justification="",
            concise_knowledge="",
        )
        self.base_model = base_model

    def __str__(self) -> str:
        if self.hypothesis is None:
            return f"No hypothesis available. Constructing first runnable {self.component} component."

        lines = [
            f"Base Model: {self.base_model}",
            f"Hypothesis: {self.hypothesis}",
        ]
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


class LLMFinetuneExpGen(ExpGen):
    """LLM fine-tuning experiment generator."""

    def __init__(self, scen: LLMFinetuneScen):
        super().__init__(scen)
        self.llama_manager = get_llama_factory_manager()

    def gen(self, trace: Trace, plan=None) -> FTExperiment:
        """Generate LLM fine-tuning experiment using LLM-driven selection."""
        # Generate hypothesis using LLM (model is always specified in current use case)
        logger.info(f"Using specified model: {FT_RD_SETTING.base_model}")
        base_model = FT_RD_SETTING.base_model
        available_models = self.llama_manager.models
        available_methods = self.llama_manager.methods

        # Generate parameter descriptions: shared params once + method-specific params
        shared_params = self.llama_manager.format_shared_params()
        methods_specific_params = {}
        for method in available_methods:
            methods_specific_params[method] = self.llama_manager.format_method_specific_params(method)

        # Prepare template context
        template_context = {
            "device_info": self.scen.device_info,
            "dataset_info": self.scen.dataset_info,
            "available_methods": ", ".join(available_methods),
            "shared_params": shared_params,
            "methods_specific_params": methods_specific_params,
            "select_model": base_model is None,
            "trace": trace,  # Pass trace object directly
            "available_models": available_models,
            "base_model": base_model,
        }

        # Use template system for prompts
        system_prompt = T(".prompts:hypothesis_gen.system_prompt").r(**template_context)
        user_prompt = T(".prompts:hypothesis_gen.user_prompt").r(**template_context)

        response_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            )
        )

        # Create hypothesis object with natural language description
        hypothesis = FTHypothesis(
            base_model=base_model,
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
        )

        ensure_ft_assets_exist(model=hypothesis.base_model, check_model=True)

        task = FTTask(base_model=hypothesis.base_model, description=response_dict.get("task"))

        return FTExperiment(sub_tasks=[task], hypothesis=hypothesis)
