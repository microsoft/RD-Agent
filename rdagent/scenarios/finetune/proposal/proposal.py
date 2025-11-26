"""LLM Fine-tuning Proposal Generator

Two-stage Decision + Router architecture:
- Stage 1: LLM decides task_type (data vs train) based on current status
- Stage 2: Route to appropriate hypothesis generation prompt

Supports two task types:
- "train": Traditional training task, generates train.yaml for LlamaFactory
- "data": Data processing task, generates main.py for COT-Self-Instruct pipeline
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

    Supports two task types:
    - "train": Hypothesis for training configuration
    - "data": Hypothesis for data processing method (COT-Self-Instruct)
    """

    def __init__(
        self,
        base_model: str,
        hypothesis: str | None = None,
        reason: str | None = None,
        task_type: Literal["train", "data"] = "train",
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

    Uses two-stage Decision + Router architecture:
    - Stage 1: LLM decides task_type based on current status
    - Stage 2: Route to appropriate hypothesis generation
    """

    def __init__(self, scen: LLMFinetuneScen):
        super().__init__(scen)

    def gen(self, trace: Trace, plan=None) -> FTExperiment:
        """Generate LLM fine-tuning experiment using two-stage architecture.

        For first loop (no history): Generate both data and train tasks sequentially
        For subsequent loops: Use decision router to choose task type
        """
        base_model = FT_RD_SETTING.base_model
        logger.info(f"Generating experiment with base model: {base_model}")

        # Check if this is the first loop (no historical experiments)
        is_first_loop = not trace.hist

        if is_first_loop:
            # First loop: Generate both data and train tasks
            logger.info("First loop detected: Generating combined data + train experiment")
            return self._gen_first_loop_experiment(trace, base_model)
        else:
            # Subsequent loops: Use decision router
            # Stage 1: Decision - Let LLM decide task_type
            task_type, decision_reason = self._decide_task_type(trace)
            logger.info(f"Stage 1 Decision: task_type = {task_type}, reason = {decision_reason}")

            # Stage 2: Router - Route to appropriate hypothesis generation
            if task_type == "data":
                return self._gen_data_hypothesis(trace, base_model, decision_reason)
            else:
                return self._gen_train_hypothesis(trace, base_model, decision_reason)

    def _gen_first_loop_experiment(self, trace: Trace, base_model: str) -> FTExperiment:
        """Generate combined experiment for first loop with both data and train tasks.

        First loop generates both:
        1. Data processing hypothesis and task (COT-Self-Instruct)
        2. Training hypothesis and task (LlamaFactory)

        Args:
            trace: Experiment trace history (should be empty for first loop)
            base_model: Base model name

        Returns:
            FTExperiment with both data and train tasks combined
        """
        logger.info("Generating first loop experiment: data processing + training")
        first_loop_reason = "First loop initialization"

        # Step 1: Generate data experiment and extract hypothesis/task
        logger.info("Step 1: Generating data processing hypothesis")
        data_exp = self._gen_data_hypothesis(trace, base_model, first_loop_reason)
        data_hypothesis = data_exp.hypothesis
        data_task = data_exp.sub_tasks[0]

        # Step 2: Generate train experiment and extract hypothesis/task
        logger.info("Step 2: Generating training hypothesis")
        train_exp = self._gen_train_hypothesis(trace, base_model, first_loop_reason)
        train_hypothesis = train_exp.hypothesis
        train_task = train_exp.sub_tasks[0]

        # Combine hypotheses into a single composite hypothesis
        combined_hypothesis = FTHypothesis(
            base_model=base_model,
            hypothesis=f"[Data Processing] {data_hypothesis.hypothesis}\n[Training] {train_hypothesis.hypothesis}",
            reason=f"First loop: Execute data processing followed by training.\n"
                   f"Data reason: {data_hypothesis.reason}\n"
                   f"Train reason: {train_hypothesis.reason}",
            task_type="data",  # Start with data type since data is processed first. Note: This is a composite task.
        )

        # Combine tasks: data task first, then train task
        combined_tasks = [data_task, train_task]

        logger.info(f"First loop experiment created with {len(combined_tasks)} tasks: "
                    f"[{', '.join(t.task_type for t in combined_tasks)}]")

        return FTExperiment(sub_tasks=combined_tasks, hypothesis=combined_hypothesis)

    def _decide_task_type(self, trace: Trace) -> tuple[str, str]:
        """Stage 1: Let LLM decide task_type based on current status.

        Provides comprehensive context to LLM:
        - Scenario description
        - Dataset status (from scen.dataset_info)
        - Detailed historical experiment results

        Returns:
            tuple: (task_type, reason)
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
        # if task_type not in ["train", "data"]:
        #     logger.warning(f"Invalid task_type '{task_type}', defaulting to 'train'")
        #     task_type = "train"
        task_type = "data"
        reason = "Because the dataset is not good enough, we need to process it."
        return task_type, reason

    def _gen_data_hypothesis(self, trace: Trace, base_model: str, decision_reason: str) -> FTExperiment:
        """Stage 2a: Generate data processing hypothesis using COT-Self-Instruct.

        Args:
            trace: Experiment trace history
            base_model: Base model name
            decision_reason: Reason from Stage 1 decision

        Returns:
            FTExperiment with data processing task
        """
        logger.info("Stage 2a: Generating data processing hypothesis")

        # Prepare parameters for prompts
        category_list = self.scen.category_dict
        dataset_folder_desc = self.scen.dataset_folder_desc

        
        # Get available finetune methods for context
        available_methods = self.llama_manager.methods

        system_prompt = T(".prompts:data_hypothesis_gen.system_prompt").r(
            scenario=self.scen.get_scenario_all_desc(),
            category_list=category_list,
            dataset_folder_desc=dataset_folder_desc,
            available_methods=available_methods,
        )
        
        user_prompt = T(".prompts:data_hypothesis_gen.user_prompt").r(
            trace=trace,
        )

        response_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            )
        )

        hypothesis = FTHypothesis(
            base_model=base_model,
            hypothesis=response_dict.get("hypothesis"),
            reason=decision_reason,
            task_type="data",
        )

        task = FTTask(
            base_model=base_model,
            description=response_dict.get("task"),
            benchmark=FT_RD_SETTING.target_benchmark,
            task_type="data",
        )

        return FTExperiment(sub_tasks=[task], hypothesis=hypothesis)

    def _gen_train_hypothesis(self, trace: Trace, base_model: str, decision_reason: str) -> FTExperiment:
        """Stage 2b: Generate training hypothesis for LlamaFactory.

        Args:
            trace: Experiment trace history
            base_model: Base model name
            decision_reason: Reason from Stage 1 decision

        Returns:
            FTExperiment with training task
        """
        logger.info("Stage 2b: Generating training hypothesis")

        # Get LlamaFactory configuration options
        available_models = self.llama_manager.models
        available_methods = self.llama_manager.methods
        shared_params = self.llama_manager.format_shared_params()

        methods_specific_params = {}
        for method in available_methods:
            methods_specific_params[method] = self.llama_manager.format_method_specific_params(method)

        system_prompt = T(".prompts:train_hypothesis_gen.system_prompt").r(
            scenario=self.scen.get_scenario_all_desc(),
            select_model=base_model is None,
            available_models=available_models,
            available_methods=available_methods,
            shared_params=shared_params,
            methods_specific_params=methods_specific_params,
        )
        user_prompt = T(".prompts:train_hypothesis_gen.user_prompt").r(
            trace=trace,
        )

        response_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            )
        )

        hypothesis = FTHypothesis(
            base_model=base_model,
            hypothesis=response_dict.get("hypothesis"),
            reason=decision_reason,
            task_type="train",
        )

        # Ensure model assets exist for training
        ensure_ft_assets_exist(model=hypothesis.base_model, check_model=True)

        task = FTTask(
            base_model=base_model,
            description=response_dict.get("task"),
            benchmark=FT_RD_SETTING.target_benchmark,
            task_type="train",
        )

        return FTExperiment(sub_tasks=[task], hypothesis=hypothesis)
