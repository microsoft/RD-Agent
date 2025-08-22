"""
LLM Fine-tuning CoSTEER Implementation

This module provides fine-tuning specific components for the CoSTEER framework,
including evaluators and evolving strategies.
"""

from pathlib import Path

import yaml

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
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
from rdagent.components.coder.data_science.share.ds_costeer import DSCoSTEER
from rdagent.components.coder.finetune.conf import FTCoderCoSTEERSettings
from rdagent.components.coder.finetune.eval import LLMFinetuneEvaluator
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.finetune.train.utils import create_parameter_validator
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent


class LLMFinetuneEvolvingStrategy(MultiProcessEvolvingStrategy):
    """LLM Fine-tuning specific evolving strategy"""

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
        base_model = getattr(target_task, "base_model", "Qwen2.5-1.5B-Instruct")
        finetune_method = getattr(target_task, "finetune_method", "lora")
        dataset = getattr(target_task, "dataset", "default")

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
        )

        # Validate the generated config using existing validator
        validated_config = self._validate_config(config_yaml)

        return {"config.yaml": validated_config}

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
    ) -> str:
        """Generate LlamaFactory configuration YAML using LLM"""

        # Prepare knowledge context
        similar_knowledge_str = ""
        if similar_knowledge:
            similar_knowledge_str = "\n".join(
                [
                    f"### Similar Implementation {i+1}:\n{knowledge.target_task.get_task_information()}\n```yaml\n{knowledge.implementation.file_dict.get('config.yaml', '')}\n```"
                    for i, knowledge in enumerate(similar_knowledge)
                ]
            )

        failed_knowledge_str = ""
        if failed_knowledge:
            failed_knowledge_str = "\n".join(
                [
                    f"### Failed Attempt {i+1}:\n```yaml\n{knowledge.implementation.file_dict.get('config.yaml', '')}\n```\n**Feedback:** {knowledge.feedback}"
                    for i, knowledge in enumerate(failed_knowledge)
                ]
            )

        # Generate prompts using templates
        system_prompt = T("components.coder.finetune.prompts:finetune_coder.system").r(
            task_desc=task_info,
            similar_knowledge=similar_knowledge_str,
            failed_knowledge=failed_knowledge_str,
        )

        user_prompt = T("components.coder.finetune.prompts:finetune_coder.user").r(
            latest_code=(
                prev_feedback.implementation.file_dict.get("config.yaml", "")
                if prev_feedback and prev_feedback.implementation
                else ""
            ),
            latest_feedback=str(prev_feedback) if prev_feedback else "",
        )

        # Call LLM to generate config
        try:
            from rdagent.oai.llm_utils import APIBackend

            api = APIBackend()
            response = api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=False,
            )

            # Extract YAML content from response
            import re

            yaml_pattern = r"```yaml\s*\n(.*?)\n```"
            match = re.search(yaml_pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

            # Fallback: try to find any YAML-like content
            yaml_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*\s*:.*?)(?=\n[a-zA-Z_][a-zA-Z0-9_]*\s*:|$)"
            matches = re.findall(yaml_pattern, response, re.DOTALL | re.MULTILINE)
            if matches:
                return "\n".join(matches)

            # If no YAML found, return fallback config
            logger.warning("No YAML configuration found in LLM response, using fallback")
            return self._generate_llamafactory_config(
                base_model, finetune_method, dataset, debug_mode, task_info, similar_knowledge, failed_knowledge
            )

        except Exception as e:
            logger.error(f"Failed to generate config with LLM: {e}, using fallback")
            return self._generate_llamafactory_config(
                base_model, finetune_method, dataset, debug_mode, task_info, similar_knowledge, failed_knowledge
            )

    def assign_code_list_to_evo(self, code_list: list[dict[str, str]], evo):
        """Assign generated code to the evolving experiment"""
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = evo.experiment_workspace
            evo.sub_workspace_list[index].inject_files(**code_list[index])
        return evo

    def _generate_llamafactory_config(
        self,
        base_model: str,
        finetune_method: str,
        dataset: str,
        debug_mode: bool = True,
        task_info: str = "",
        similar_knowledge: list = None,
        failed_knowledge: list = None,
    ) -> str:
        """Generate LlamaFactory configuration YAML"""

        # Base configuration
        config = {
            "model_name_or_path": base_model,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": finetune_method,
            "dataset": dataset,
            "template": "qwen",
            "cutoff_len": 2048,
            "max_samples": 100 if debug_mode else None,  # Debug mode uses only 100 samples
            "overwrite_cache": True,
            "preprocessing_num_workers": 16,
            "output_dir": "./saves",
            "logging_steps": 10,
            "save_steps": 500,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "num_train_epochs": 3.0,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "bf16": True,
            "ddp_timeout": 180000000,
            "val_size": 0.1,
            "eval_strategy": "steps",
            "eval_steps": 250,
            "per_device_eval_batch_size": 1,
            "seed": 42,
        }

        # Add method-specific parameters
        if finetune_method in ["lora", "qlora"]:
            config.update(
                {
                    "lora_rank": 8,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1,
                    "lora_target": "all",
                }
            )

        if finetune_method == "qlora":
            config.update(
                {
                    "quantization_bit": 4,
                    "double_quantization": True,
                    "quantization_type": "nf4",
                }
            )

        # Convert to YAML string
        yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)

        return yaml_content

    def _validate_config(self, config_yaml: str) -> str:
        """Validate LlamaFactory configuration using existing validator"""
        try:
            return self.parameter_validator.validate_yaml_config(config_yaml)
        except Exception as e:
            from rdagent.log import rdagent_logger as logger

            logger.warning(f"Config validation failed: {e}, using original config")
            return config_yaml


class LLMFinetuneCoSTEER(DSCoSTEER):
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

        # Initialize LlamaFactory parameter validator
        self.parameter_validator = create_parameter_validator()
