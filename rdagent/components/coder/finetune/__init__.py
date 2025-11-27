"""
LLM Fine-tuning CoSTEER Implementation

This module provides fine-tuning specific components for the CoSTEER framework,
including evaluators and evolving strategies.
"""

import re
from pathlib import Path
from typing import Callable

import yaml

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiEvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.components.coder.finetune.conf import (
    FT_YAML_FILE_NAME,
    FTCoderCoSTEERSettings,
)
from rdagent.components.coder.finetune.eval import FTCoderEvaluator, FTDataEvaluator
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.finetune.scen.llama_factory_manager import LLaMAFactory_manager
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent


class LLMFinetuneEvolvingStrategy(MultiProcessEvolvingStrategy):
    """LLM Fine-tuning specific evolving strategy"""

    def __init__(self, scen: Scenario, settings, *args, **kwargs):
        super().__init__(scen, settings)
        self.llama_factory_manager = LLaMAFactory_manager

    def implement_func_list(self) -> list[Callable]:
        return [self.implement_data, self.implement_lf_config]

    def implement_data(
            self,
            target_task: Task,
            queried_knowledge: CoSTEERQueriedKnowledge | None = None,
            workspace: FBWorkspace | None = None,
            prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        return {}

    def implement_lf_config(
        self,
        target_task: Task,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        """Implement a single fine-tuning task by generating LlamaFactory config"""

        task_info = target_task.get_task_information()

        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[task_info] if queried_knowledge is not None else []
        )
        queried_former_failed_knowledge = (
            [
                knowledge
                for knowledge in queried_former_failed_knowledge[0]
                if knowledge.implementation.file_dict.get(FT_YAML_FILE_NAME)
                != workspace.file_dict.get(FT_YAML_FILE_NAME)
            ],
            queried_former_failed_knowledge[1],
        )

        # Get task parameters from the task object
        base_model = getattr(target_task, "base_model")

        # Use LLM to generate LlamaFactory config YAML
        # Coder will decide method based on hypothesis and available parameters
        config_yaml = self._generate_llamafactory_config_with_llm(
            base_model=base_model,
            task_info=task_info,
            queried_former_failed_knowledge=queried_former_failed_knowledge,
            prev_feedback=prev_task_feedback,
            workspace=workspace,
        )

        # Return generated config directly - validation happens in evaluator
        return {FT_YAML_FILE_NAME: config_yaml}

    def _generate_llamafactory_config_with_llm(
        self,
        base_model: str,
        task_info: str = "",
        queried_former_failed_knowledge: tuple = None,
        prev_feedback=None,
        workspace=None,
    ) -> str:
        """Generate LlamaFactory configuration YAML using LLM"""

        # Query LLaMA Factory parameters: shared params once + method-specific params
        available_methods = self.llama_factory_manager.methods
        shared_params = self.llama_factory_manager.format_shared_params()

        # Format method-specific parameters only (no duplication of shared params)
        methods_specific_params = {}
        for method in available_methods:
            methods_specific_params[method] = self.llama_factory_manager.format_method_specific_params(method)

        # Use fixed Docker paths for simplicity
        models_path = "/assets/models/"
        datasets_path = "/assets/datasets/"

        # Generate prompts using templates with all required parameters
        system_prompt = T(".prompts:finetune_coder.system").r(
            scenario=self.scen.get_scenario_all_desc(),
            task_desc=task_info,
            queried_former_failed_knowledge=queried_former_failed_knowledge[0],
            available_methods=", ".join(available_methods),
            shared_params=shared_params,
            methods_specific_params=methods_specific_params,
        )

        user_prompt = T(".prompts:finetune_coder.user").r(
            latest_code=workspace.file_dict.get(FT_YAML_FILE_NAME, ""),
            latest_feedback=prev_feedback,
            base_model=base_model,
            models_path=models_path,
            datasets_path=datasets_path,
        )

        # Call LLM to generate config
        try:
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=False,
            )

            # Extract YAML content from response
            # Try markdown code block first (standard format from improved prompt)
            match = re.search(r"```(?:yaml)?\s*\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_yaml = match.group(1).strip()
                try:
                    yaml.safe_load(extracted_yaml)
                    logger.info("Extracted YAML from markdown code block")
                    return extracted_yaml
                except yaml.YAMLError as e:
                    logger.warning(f"Extracted YAML is invalid: {e}")
                    raise RuntimeError(f"Invalid YAML in code block: {e}")

            # Fallback: try to use entire response as YAML
            try:
                yaml.safe_load(response)
                logger.info("Using entire response as YAML")
                return response.strip()
            except yaml.YAMLError as e:
                logger.error(f"Failed to parse response as YAML: {e}")
                raise RuntimeError(f"Failed to extract valid YAML from LLM response: {e}")

        except Exception as e:
            logger.error(f"Failed to generate config with LLM: {e}")
            raise RuntimeError(f"LLM config generation failed: {e}")

    def assign_code_list_to_evo(self, code_list: list[dict[str, str]], evo):
        """Assign generated code to the evolving experiment"""
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = evo.experiment_workspace
            evo.sub_workspace_list[index].inject_files(**code_list[index])
        return evo


class LLMFinetuneCoSTEER(CoSTEER):
    """LLM Fine-tuning CoSTEER implementation"""

    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        settings = FTCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator([FTDataEvaluator(scen=scen), FTCoderEvaluator(scen=scen)], scen=scen)
        es = LLMFinetuneEvolvingStrategy(scen=scen, settings=settings)

        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            max_loop=FT_RD_SETTING.coder_max_loop if hasattr(FT_RD_SETTING, "coder_max_loop") else 5,
            stop_eval_chain_on_fail=True, # finetune involve partial implementation.
            **kwargs,
        )
