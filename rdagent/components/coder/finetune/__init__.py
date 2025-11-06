"""
LLM Fine-tuning CoSTEER Implementation

This module provides fine-tuning specific components for the CoSTEER framework,
including evaluators and evolving strategies.
"""

import re
from pathlib import Path

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
from rdagent.components.coder.finetune.conf import FTCoderCoSTEERSettings
from rdagent.components.coder.finetune.eval import FTCoderEvaluator
from rdagent.components.coder.finetune.exp import TrainingTask
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent
FT_YAML_FILE_NAME = "train.yaml"


class LLMFinetuneEvolvingStrategy(MultiProcessEvolvingStrategy):
    """LLM Fine-tuning specific evolving strategy"""

    def __init__(self, scen: Scenario, settings, *args, **kwargs):
        super().__init__(scen, settings)

        # Lazy import to avoid circular dependency
        from rdagent.scenarios.finetune.scen.llama_factory_manager import (
            get_llama_factory_manager,
        )

        self.llama_factory_manager = get_llama_factory_manager()

    def implement_one_task(
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
        finetune_method = getattr(target_task, "finetune_method")
        dataset = getattr(target_task, "dataset")

        # Use LLM to generate LlamaFactory config YAML
        # Generate full training configuration - validator will test with micro-batch automatically
        config_yaml = self._generate_llamafactory_config_with_llm(
            base_model=base_model,
            finetune_method=finetune_method,
            dataset=dataset,
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
        finetune_method: str,
        dataset: str,
        task_info: str = "",
        queried_former_failed_knowledge: tuple = None,
        prev_feedback=None,
        workspace=None,
    ) -> str:
        """Generate LlamaFactory configuration YAML using LLM"""

        # Query LLaMA Factory parameters for the specific method
        method_params_desc = self.llama_factory_manager.format_method_params(finetune_method)

        # Use fixed Docker paths for simplicity
        models_path = "/assets/models/"
        datasets_path = "/assets/datasets/"

        # Generate prompts using templates with all required parameters
        # TODO: give exp_gen(natural language) here
        system_prompt = T("components.coder.finetune.prompts:finetune_coder.system").r(
            task_desc=task_info,
            finetune_method=finetune_method,
            queried_former_failed_knowledge=queried_former_failed_knowledge[0],
            method_params=method_params_desc,
        )

        user_prompt = T("components.coder.finetune.prompts:finetune_coder.user").r(
            latest_code=workspace.file_dict.get(FT_YAML_FILE_NAME, ""),
            latest_feedback=prev_feedback,
            finetune_method=finetune_method,
            base_model=base_model,
            dataset_name=dataset,
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
        eva = CoSTEERMultiEvaluator(FTCoderEvaluator(scen=scen), scen=scen)
        es = LLMFinetuneEvolvingStrategy(scen=scen, settings=settings)

        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            max_loop=FT_RD_SETTING.coder_max_loop if hasattr(FT_RD_SETTING, "coder_max_loop") else 5,
            **kwargs,
        )
