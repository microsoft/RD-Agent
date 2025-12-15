"""
LLM Fine-tuning CoSTEER Implementation

This module provides fine-tuning specific components for the CoSTEER framework,
including evaluators and evolving strategies.
"""

import json
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
    FT_DATA_FILE_NAME,
    FT_DATA_SCRIPT_NAME,
    FT_PATHS,
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
        """Generate data processing script based on task.

        This method generates a Python script that processes seed datasets
        and outputs a data.json file in Alpaca format.

        Returns:
            dict with "process_data.py" key containing the script code,
            or empty dict if data already exists.
        """
        # Check if data.json already exists and evaluation passed
        if workspace is not None:
            data_json_path = workspace.workspace_path / FT_DATA_FILE_NAME
            if data_json_path.exists():
                # Only skip if no prior feedback (first run success) or evaluation passed
                if prev_task_feedback is None or prev_task_feedback.final_decision:
                    logger.info("data.json already exists and passed evaluation, skipping")
                    return {}
                # Evaluation failed, remove old data and regenerate
                logger.info("data.json exists but evaluation failed, regenerating script...")
                data_json_path.unlink()

        # Get dataset information for the task
        involving_datasets = getattr(target_task, "involving_datasets", [])
        dataset_info = self._get_dataset_info(involving_datasets)

        # Generate data processing script using LLM
        system_prompt = T(".prompts:data_coder.system").r(
            scenario=self.scen.get_scenario_all_desc(),
            task_desc=target_task.get_task_information(),
            dataset_info=dataset_info,
            prev_feedback=prev_task_feedback,
            api_max_workers=FT_RD_SETTING.api_max_workers,
            datasets_path=FT_PATHS.datasets,
            workspace_path=FT_PATHS.workspace,
        )

        user_prompt = T(".prompts:data_coder.user").r(
            datasets_path=FT_PATHS.datasets,
            workspace_path=FT_PATHS.workspace,
        )

        try:
            script_code = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=False,
                code_block_language="python",
                code_block_fallback=False,
            )
            logger.info(f"Generated data processing script ({len(script_code)} chars)")

            return {FT_DATA_SCRIPT_NAME: script_code}

        except Exception as e:
            logger.error(f"Failed to generate data processing script: {e}")
            raise RuntimeError(f"Data processing script generation failed: {e}")

    def _get_dataset_info(self, involving_datasets: list[str]) -> str:
        """Read dataset_info.json and return information for specified datasets."""
        datasets_dir = Path(FT_RD_SETTING.file_path) / "datasets"
        dataset_info_path = datasets_dir / "dataset_info.json"

        if not dataset_info_path.exists():
            logger.warning(f"dataset_info.json not found at {dataset_info_path}")
            return "No dataset information available."

        try:
            with open(dataset_info_path, "r", encoding="utf-8") as f:
                all_dataset_info = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read dataset_info.json: {e}")
            return f"Error reading dataset info: {e}"

        # Filter to only involved datasets, or use all if none specified
        if involving_datasets:
            filtered_info = {name: info for name, info in all_dataset_info.items() if name in involving_datasets}
        else:
            filtered_info = all_dataset_info

        if not filtered_info:
            return "No matching datasets found in dataset_info.json."

        # Format dataset info for the prompt
        info_parts = []
        for name, info in filtered_info.items():
            info_text = f"### Dataset: {name}\n"
            info_text += f"- File: {info.get('file_name', 'N/A')}\n"
            info_text += f"- Format: {info.get('formatting', 'N/A')}\n"

            if info.get("columns"):
                info_text += f"- Columns: {json.dumps(info['columns'])}\n"

            if info.get("description"):
                # Truncate long descriptions
                desc = info["description"]
                if len(desc) > 500:
                    desc = desc[:500] + "..."
                info_text += f"- Description: {desc}\n"

            if info.get("stats"):
                stats = info["stats"]
                info_text += f"- Sample count: {stats.get('sample_count', 'N/A')}\n"

            info_parts.append(info_text)

        return "\n".join(info_parts)

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

        # Use environment-aware paths (Docker vs Conda)
        # Note: datasets_path in finetune_coder uses workspace path where processed
        # data.json and dataset_info.json are located (generated by FTDataEvaluator)

        # Generate prompts using templates with all required parameters
        system_prompt = T(".prompts:finetune_coder.system").r(
            scenario=self.scen.get_scenario_all_desc(),
            task_desc=task_info,
            queried_former_failed_knowledge=queried_former_failed_knowledge[0],
            available_methods=", ".join(available_methods),
            shared_params=shared_params,
            methods_specific_params=methods_specific_params,
        )

        # Read data_stats.json from workspace (injected by FTDataEvaluator)
        data_stats = workspace.file_dict.get("data_stats.json", "")

        user_prompt = T(".prompts:finetune_coder.user").r(
            latest_code=workspace.file_dict.get(FT_YAML_FILE_NAME, ""),
            latest_feedback=prev_feedback,
            base_model=base_model,
            models_path=FT_PATHS.models,
            datasets_path=FT_PATHS.workspace,  # Training config uses workspace path for processed data
            workspace_path=FT_PATHS.workspace,
            deepspeed_path=FT_PATHS.deepspeed,  # Empty string in conda mode (disables DeepSpeed hints)
            data_stats=data_stats,
        )

        # Call LLM to generate config
        try:
            extracted_yaml = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=False,
                code_block_language="yaml",
                code_block_fallback=False, 
            )

            # Validate YAML syntax
            try:
                yaml.safe_load(extracted_yaml)
                logger.info("Extracted YAML config successfully")
                return extracted_yaml
            except yaml.YAMLError as e:
                logger.error(f"Invalid YAML syntax: {e}")
                raise RuntimeError(f"Invalid YAML syntax: {e}")

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
            stop_eval_chain_on_fail=True,  # finetune involve partial implementation.
            **kwargs,
        )
