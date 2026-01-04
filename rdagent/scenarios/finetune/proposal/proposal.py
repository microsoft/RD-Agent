"""LLM Fine-tuning Proposal Generator

Unified hypothesis generation that covers both data processing and training configuration.
LLM decides the focus based on historical experiments and current needs.
"""

import json

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.exp import FTTask
from rdagent.core.proposal import ExpGen, Hypothesis, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.finetune.experiment.experiment import FTExperiment
from rdagent.scenarios.finetune.proposal.trace import FTTrace
from rdagent.scenarios.finetune.scen.llama_factory_manager import (
    LLaMAFactory_manager,
)
from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen
from rdagent.scenarios.finetune.utils import ensure_ft_assets_exist
from rdagent.utils.agent.tpl import T


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
    """LLM fine-tuning experiment generator.

    Generates unified hypothesis covering both data processing and training configuration.
    """

    def __init__(self, scen: LLMFinetuneScen):
        super().__init__(scen)

    def gen(self, trace: Trace, plan=None) -> FTExperiment:
        """Generate LLM fine-tuning experiment."""
        base_model = FT_RD_SETTING.base_model
        logger.info(f"Generating experiment with base model: {base_model}")

        is_first_loop = not trace.hist
        return self._gen_hypothesis(trace, base_model, is_first_loop)

    def _gen_hypothesis(self, trace: Trace, base_model: str, is_first_loop: bool) -> FTExperiment:
        """Generate hypothesis covering both data processing and training configuration.

        Args:
            trace: Experiment trace history
            base_model: Base model name
            is_first_loop: Whether this is the first experiment

        Returns:
            FTExperiment with tasks for both data processing and training
        """
        logger.info(f"Generating hypothesis (is_first_loop={is_first_loop})")

        available_models = LLaMAFactory_manager.models
        available_methods = LLaMAFactory_manager.methods
        shared_params = LLaMAFactory_manager.format_shared_params()
        methods_specific_params = {}
        for method in available_methods:
            methods_specific_params[method] = LLaMAFactory_manager.format_method_specific_params(method)

        # Get SOTA experiment info for guidance
        sota_info = trace.sota_info() if isinstance(trace, FTTrace) else None
        if sota_info:
            logger.info("SOTA experiment found, will include in hypothesis generation")

        system_prompt = T(".prompts:unified_hypothesis_gen.system_prompt").r(
            is_first_loop=is_first_loop,
            scenario=self.scen.get_scenario_all_desc(enable_dataset_description=True),
            available_models=available_models,
            available_methods=available_methods,
            shared_params=shared_params,
            methods_specific_params=methods_specific_params,
            select_model=base_model is None,
        )

        user_prompt = T(".prompts:unified_hypothesis_gen.user_prompt").r(
            trace=trace,
            sota_info=sota_info,
        )

        session = APIBackend().build_chat_session(session_system_prompt=system_prompt)
        reason_dict = json.loads(
            session.build_chat_completion(
                user_prompt=user_prompt + "\n" + T(".prompts:unified_hypothesis_gen.specific_format").r(field="reason"),
                json_target_type=dict,
            )
        )
        hypothesis_dict = json.loads(
            session.build_chat_completion(
                user_prompt=T(".prompts:unified_hypothesis_gen.specific_format").r(field="hypothesis"),
                json_target_type=dict,
            )
        )
        task_dict = json.loads(
            session.build_chat_completion(
                user_prompt=T(".prompts:unified_hypothesis_gen.specific_format").r(field="task"),
                json_target_type=dict,
            )
        )

        ensure_ft_assets_exist(model=base_model, check_model=True)

        # Get skip_data_processing from task_dict (merged with task in 3rd LLM call)
        # Only valid for subsequent experiments, first experiment always generates data
        skip_data_processing = task_dict.get("skip_data_processing", False) if not is_first_loop else False
        if skip_data_processing:
            logger.info("Proposal decided to skip data processing, will reuse SOTA's data.json")

        # Use pre-selected datasets from scenario initialization
        task = FTTask(
            base_model=base_model,
            description=task_dict.get("task"),
            benchmark=FT_RD_SETTING.target_benchmark,
            involving_datasets=self.scen.selected_datasets,
            skip_data_processing=skip_data_processing,
        )

        hypothesis = FTHypothesis(
            base_model=base_model,
            hypothesis=hypothesis_dict.get("hypothesis"),
            reason=reason_dict.get("reason", ""),
        )

        exp = FTExperiment(sub_tasks=[task], hypothesis=hypothesis)

        # Inject workspace files from SOTA experiment (if available)
        if isinstance(trace, FTTrace):
            sota_exp = trace.sota_experiment()
            if sota_exp is not None and (ws := sota_exp.experiment_workspace) is not None and ws.file_dict:
                exp.experiment_workspace.inject_code_from_file_dict(ws)
                logger.info(f"Injected {len(ws.file_dict)} files from SOTA: {list(ws.file_dict.keys())}")

        logger.info("Experiment created")

        return exp
