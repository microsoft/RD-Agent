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
from rdagent.components.coder.finetune.eval import LLMFinetuneEvaluator
from rdagent.components.coder.finetune.exp import TrainingTask
from rdagent.components.coder.finetune.unified_validator import create_unified_validator
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent


class LLMFinetuneEvolvingStrategy(MultiProcessEvolvingStrategy):
    """LLM Fine-tuning specific evolving strategy"""

    def __init__(self, scen: Scenario, settings, *args, **kwargs):
        super().__init__(scen, settings)

        self.config_validator = create_unified_validator()

    def implement_one_task(
        self,
        target_task: Task,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        """Implement a single fine-tuning task by generating LlamaFactory config"""

        task_info = target_task.get_task_information()

        # Query relevant knowledge
        similar_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge.get(task_info, []) if queried_knowledge else []
        )

        failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces.get(task_info, ([], None))
            if queried_knowledge
            else ([], None)
        )

        # Get task parameters from the task object
        base_model = getattr(target_task, "base_model")
        finetune_method = getattr(target_task, "finetune_method")
        dataset = getattr(target_task, "dataset")

        # Use LLM to generate LlamaFactory config YAML
        # For coding stage, use debug mode with limited samples
        config_yaml = self._generate_llamafactory_config_with_llm(
            base_model=base_model,
            finetune_method=finetune_method,
            dataset=dataset,
            debug_mode=True,  # Use debug mode for coding stage (limited samples for quick validation)
            task_info=task_info,
            similar_knowledge=similar_knowledge,
            failed_knowledge=failed_knowledge[0],
            prev_feedback=prev_task_feedback,
            workspace=workspace,
        )

        # Validate the generated config using existing validator
        validated_config = self._validate_config(config_yaml)

        return {"train.yaml": validated_config}

    def _generate_llamafactory_config_with_llm(
        self,
        base_model: str,
        finetune_method: str,
        dataset: str,
        debug_mode: bool = True,
        task_info: str = "",
        similar_knowledge: list = None,
        failed_knowledge: list = None,
        prev_feedback=None,
        workspace=None,
    ) -> str:
        """Generate LlamaFactory configuration YAML using LLM"""

        # Prepare knowledge context
        similar_knowledge_str = ""
        if similar_knowledge:
            similar_knowledge_str = "\n".join(
                [
                    f"### Similar Implementation {i+1}:\n{knowledge.target_task.get_task_information()}\n```yaml\n{knowledge.implementation.file_dict.get('train.yaml', '')}\n```"
                    for i, knowledge in enumerate(similar_knowledge)
                ]
            )

        failed_knowledge_str = ""
        if failed_knowledge and isinstance(failed_knowledge, (list, tuple)) and len(failed_knowledge) > 0:
            # Handle both list of knowledge and tuple format (knowledge_list, None)
            knowledge_list = failed_knowledge[0] if isinstance(failed_knowledge, tuple) else failed_knowledge
            if knowledge_list:
                failed_knowledge_str = "\n".join(
                    [
                        f"### Failed Attempt {i+1}:\n```yaml\n{knowledge.implementation.file_dict.get('train.yaml', '')}\n```\n**Feedback:** {knowledge.feedback}"
                        for i, knowledge in enumerate(knowledge_list)
                    ]
                )

        # Query LLaMA Factory parameters for the specific method
        from rdagent.scenarios.finetune.llama_factory_manager import LLaMAFactoryManager

        llama_manager = LLaMAFactoryManager()
        method_params_desc = llama_manager.format_method_params(finetune_method)

        # Use fixed Docker paths for simplicity
        models_path = "/assets/models/"
        datasets_path = "/assets/datasets/"

        # Generate prompts using templates with all required parameters
        system_prompt = T("components.coder.finetune.prompts:finetune_coder.system").r(
            task_desc=task_info,
            finetune_method=finetune_method,
            similar_knowledge=similar_knowledge if similar_knowledge else [],
            failed_knowledge=failed_knowledge[0] if isinstance(failed_knowledge, tuple) and failed_knowledge[0] else [],
            method_params=method_params_desc,
        )

        user_prompt = T("components.coder.finetune.prompts:finetune_coder.user").r(
            latest_code=(workspace.file_dict.get("train.yaml", "") if workspace and prev_feedback else ""),
            latest_feedback=str(prev_feedback) if prev_feedback else "",
            finetune_method=finetune_method,
            base_model=base_model,
            dataset_name=dataset,
            models_path=models_path,
            datasets_path=datasets_path,
        )

        # Call LLM to generate config
        try:
            api = APIBackend()
            response = api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=False,
            )

            # Extract YAML content from response
            # Try multiple YAML extraction patterns
            yaml_patterns = [
                r"```yaml\s*\n(.*?)\n```",  # Standard markdown yaml block
                r"```\s*\n(.*?)\n```",  # Generic code block
                r"yaml\s*:\s*\n(.*?)(?=\n\S|\Z)",  # yaml: prefix
            ]

            extracted_yaml = None
            for pattern in yaml_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted_yaml = match.group(1).strip()
                    logger.info(f"Extracted YAML using pattern: {pattern}")
                    break

            if extracted_yaml:
                # Validate extracted YAML
                try:
                    yaml.safe_load(extracted_yaml)
                    return extracted_yaml
                except yaml.YAMLError as e:
                    logger.warning(f"Extracted YAML is invalid: {e}, trying fallback")

            # Fallback: try to find any YAML-like content
            yaml_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*\s*:.*?)(?=\n[a-zA-Z_][a-zA-Z0-9_]*\s*:|$)"
            matches = re.findall(yaml_pattern, response, re.DOTALL | re.MULTILINE)
            if matches:
                fallback_yaml = "\n".join(matches)
                try:
                    yaml.safe_load(fallback_yaml)
                    return fallback_yaml
                except yaml.YAMLError:
                    logger.warning("Fallback YAML extraction also failed")

            # If no YAML found, raise error
            logger.error("No YAML configuration found in LLM response")
            raise RuntimeError("Failed to extract valid YAML from LLM response")

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

    def _validate_config(self, config_yaml: str) -> str:
        """Validate configuration using unified validator"""
        try:
            result = self.config_validator.validate_config_comprehensive(
                config_yaml, enable_micro_batch_test=False  # Skip micro-batch test in coder stage for speed
            )

            report = self.config_validator.generate_validation_report(result)
            logger.info(f"Config validation:\n{report}")

            if not result.success:
                logger.warning(f"Validation failed: {result.errors}")

            return result.filtered_config  # Always return filtered config

        except Exception as e:
            logger.warning(f"Config validation exception: {e}")
            return config_yaml


class LLMFinetuneCoSTEER(CoSTEER):
    """LLM Fine-tuning CoSTEER implementation"""

    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        settings = FTCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator(LLMFinetuneEvaluator(scen=scen), scen=scen)
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
