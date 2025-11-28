"""LLM Fine-tuning Proposal Generator

Two-stage Decision + Generation architecture:
- Stage 1: LLM decides task_type (data/train/both) based on current status
- Stage 2: Unified hypothesis generation based on task_type

Supports three task types:
- "train": Training task only, generates train.yaml for LlamaFactory
- "data": Data processing task only, generates data processing code
- "both": Combined task, generates both data processing and training configurations
"""

import json
from typing import Literal

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.exp import FTTask
from rdagent.core.proposal import ExpGen, Hypothesis, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.finetune.experiment.experiment import FTExperiment
from rdagent.scenarios.finetune.scen.llama_factory_manager import (
    LLaMAFactory_manager,
)
from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen
from rdagent.scenarios.finetune.utils import ensure_ft_assets_exist
from rdagent.utils.agent.tpl import T


class FTHypothesis(Hypothesis):
    """LLM fine-tuning hypothesis class.

    Supports three task types:
    - "train": Hypothesis for training configuration
    - "data": Hypothesis for data processing method
    - "both": Combined hypothesis for both data processing and training
    """

    def __init__(
        self,
        base_model: str,
        hypothesis: str | None = None,
        reason: str | None = None,
        task_type: Literal["train", "data", "both"] = "train",
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
        self.task_type = task_type

    def __str__(self) -> str:
        if self.hypothesis is None:
            return f"No hypothesis available. Constructing first runnable {self.component} component."

        lines = [
            f"Task Type: {self.task_type}",
            f"Base Model: {self.base_model}",
            f"Hypothesis: {self.hypothesis}",
        ]
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


class LLMFinetuneExpGen(ExpGen):
    """LLM fine-tuning experiment generator.

    Uses two-stage Decision + Generation architecture:
    - Stage 1: LLM decides task_type (data/train/both) based on current status
    - Stage 2: Unified hypothesis generation based on task_type
    """

    def __init__(self, scen: LLMFinetuneScen):
        super().__init__(scen)

    def gen(self, trace: Trace, plan=None) -> FTExperiment:
        """Generate LLM fine-tuning experiment using two-stage architecture.

        Stage 1: Decide task_type
            - First loop: task_type = "both" (generate data + train tasks)
            - Subsequent loops: LLM decides task_type from ["data", "train", "both"]

        Stage 2: Generate hypothesis and tasks based on task_type
        """
        base_model = FT_RD_SETTING.base_model
        logger.info(f"Generating experiment with base model: {base_model}")

        # Check if this is the first loop (no historical experiments)
        is_first_loop = not trace.hist

        if is_first_loop:
            # First loop: Generate both data and train tasks
            logger.info("First loop detected: Generating combined data + train experiment")
            task_type = "both"
            decision_reason = "First loop initialization"
        else:
            # Subsequent loops: Use decision router
            # Stage 1: Decision - Let LLM decide task_type
            task_type, decision_reason = self._decide_task_type(trace)
            logger.info(f"Stage 1 Decision: task_type = {task_type}, reason = {decision_reason}")

        # Stage 2: Generate hypothesis based on task_type
        return self._gen_hypothesis(trace, base_model, task_type, decision_reason)

    def _gen_hypothesis(
        self, trace: Trace, base_model: str, task_type: Literal["data", "train", "both"], decision_reason: str
    ) -> FTExperiment:
        """Unified hypothesis generation method.

        Args:
            trace: Experiment trace history
            base_model: Base model name
            task_type: Type of task to generate ("data", "train", or "both")
            decision_reason: Reason for this task type selection

        Returns:
            FTExperiment with appropriate task(s) based on task_type
        """
        logger.info(f"Generating hypothesis for task_type: {task_type}")

        # Prepare all necessary context for prompts
        available_models = LLaMAFactory_manager.models
        available_methods = LLaMAFactory_manager.methods
        shared_params = LLaMAFactory_manager.format_shared_params()
        methods_specific_params = {}
        for method in available_methods:
            methods_specific_params[method] = LLaMAFactory_manager.format_method_specific_params(method)


        # Build unified prompt with task_type
        system_prompt = T(".prompts:unified_hypothesis_gen.system_prompt").r(
            task_type=task_type,
            scenario=self.scen.get_scenario_all_desc(enable_dataset_description=True),
            available_models=available_models,
            available_methods=available_methods,
            shared_params=shared_params,
            methods_specific_params=methods_specific_params,
            select_model=base_model is None,
        )

        user_prompt = T(".prompts:unified_hypothesis_gen.user_prompt").r(
            trace=trace,
            task_type=task_type,
        )

        # Single LLM call - prompt decides what to generate based on task_type
        response_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_target_type=dict,
            )
        )

        # Parse involving_datasets for data/both task types
        involving_datasets = []
        if task_type in ["data", "both"]:
            involving_datasets_raw = response_dict.get("involving_datasets", "[]")
            if isinstance(involving_datasets_raw, str):
                cleaned = involving_datasets_raw.strip().strip("[]")
                involving_datasets = [ds.strip() for ds in cleaned.split(",") if ds.strip()]
            else:
                involving_datasets = involving_datasets_raw if involving_datasets_raw else []

        # Ensure model assets exist for training tasks
        if task_type in ["train", "both"]:
            ensure_ft_assets_exist(model=base_model, check_model=True)

        # Create single task with appropriate task_type
        task = FTTask(
            base_model=base_model,
            description=response_dict.get("task"),
            benchmark=FT_RD_SETTING.target_benchmark,
            task_type=task_type,
            involving_datasets=involving_datasets,
        )

        # Create hypothesis from response
        hypothesis = FTHypothesis(
            base_model=base_model,
            hypothesis=response_dict.get("hypothesis"),
            reason=f"{decision_reason}\n{response_dict.get('reason', '')}",
            task_type=task_type,
        )

        logger.info(f"Experiment created with task_type: {task_type}")

        return FTExperiment(sub_tasks=[task], hypothesis=hypothesis)

    def _decide_task_type(self, trace: Trace) -> tuple[Literal["data", "train", "both"], str]:
        """Stage 1: Let LLM decide task_type based on current status.

        Provides comprehensive context to LLM:
        - Scenario description
        - Dataset status (from scen.dataset_info)
        - Detailed historical experiment results

        Returns:
            tuple: (task_type, reason) where task_type can be "data", "train", or "both"
        """
        # TODO: given the feedback to choose whether to do data processing or training
        # # Gather dataset status information
        # dataset_info = getattr(self.scen, "dataset_info", None)

        # system_prompt = T(".prompts:task_type_decision.system_prompt").r(
        #     scenario=self.scen.get_scenario_all_desc(),
        #     dataset_info=dataset_info,
        # )
        # user_prompt = T(".prompts:task_type_decision.user_prompt").r(
        #     trace=trace,
        # )

        # logger.info(f"Stage 1: Deciding task type with {len(trace.hist) if trace.hist else 0} historical experiments")

        # response = APIBackend().build_messages_and_create_chat_completion(
        #     user_prompt=user_prompt,
        #     system_prompt=system_prompt,
        #     json_mode=True,
        # )

        # result = json.loads(response)
        # task_type = result.get("task_type", "train")
        # reason = result.get("reason", "")

        # # Validate task_type
        # if task_type not in ["train", "data", "both"]:
        #     logger.warning(f"Invalid task_type '{task_type}', defaulting to 'train'")
        #     task_type = "train"
        task_type = "data"
        reason = "Because the dataset is not good enough, we need to process it."
        return task_type, reason
