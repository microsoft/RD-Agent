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
    FT_DATA_SCRIPT_NAME,
    FT_PATHS,
    FT_TEST_PARAMS_FILE_NAME,
    FT_YAML_FILE_NAME,
    FTCoderCoSTEERSettings,
)
from rdagent.components.coder.finetune.eval import FTCoderEvaluator, FTDataEvaluator
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.finetune.scen.llama_factory_manager import LLaMAFactory_manager
from rdagent.scenarios.finetune.scen.utils import FinetuneDatasetDescriptor
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
        # Check if proposal decided to skip data processing (reuse SOTA's data processing script)
        if getattr(target_task, "skip_data_processing", False):
            # Defensive check: ensure data script actually exists before skipping
            script_exists = False
            if workspace is not None:
                script_exists = FT_DATA_SCRIPT_NAME in workspace.file_dict

            if script_exists:
                logger.info("Proposal decided to skip data processing, reusing SOTA's data script")
                return {}
            else:
                logger.warning(
                    "skip_data_processing=True but process_data.py not found in workspace, "
                    "this indicates SOTA injection failed - system design issue"
                )
                # Don't fallback silently, let it fail early to expose the issue

        # check whether the current code passes evaluation
        if (
            prev_task_feedback is not None
            and "FTDataEvaluator" in prev_task_feedback.source_feedback
            and prev_task_feedback.source_feedback["FTDataEvaluator"]
        ):
            logger.info("Previous data processing code passed evaluation, skipping regeneration")
            return {}

        # build former failed trace
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[target_task.get_task_information()]
            if queried_knowledge is not None
            else []
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

        # Get dataset information for the task
        involving_datasets = getattr(target_task, "involving_datasets", [])
        dataset_info = self._get_dataset_info(involving_datasets, datasets_path=FT_PATHS.datasets)

        # Generate data processing script using LLM
        system_prompt = T(".prompts:data_coder.system").r(
            scenario=self.scen.get_scenario_all_desc(),
            task_desc=target_task.get_task_information(),
            dataset_info=dataset_info,
            queried_former_failed_knowledge=queried_former_failed_knowledge[0],
            api_max_workers=FT_RD_SETTING.api_max_workers,
            datasets_path=FT_PATHS.datasets,
            workspace_path=FT_PATHS.workspace,
            force_think_token=FT_RD_SETTING.force_think_token,
        )

        user_prompt = T(".prompts:data_coder.user").r(
            datasets_path=FT_PATHS.datasets,
            workspace_path=FT_PATHS.workspace,
            latest_code=workspace.file_dict.get(FT_DATA_SCRIPT_NAME, "") if workspace else "",
            latest_feedback=prev_task_feedback,
            involved_dataset_folder_desc={
                ds_name: FinetuneDatasetDescriptor().describe_dataset_folder(
                    Path(FT_RD_SETTING.file_path) / "datasets" / ds_name, include_dataset_readme=True
                )
                for ds_name in involving_datasets
            },
        )

        script_code = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
            code_block_language="python",
            code_block_fallback=False,
        )
        logger.info(f"Generated data processing script ({len(script_code)} chars)")

        return {FT_DATA_SCRIPT_NAME: script_code}

    def _get_dataset_info(self, involving_datasets: list[str], datasets_path: str = None) -> str:
        """Read dataset_info.json and return information for specified datasets.

        Handles unified tasks structure:
        - readme: Dataset README content
        - file_tree: Directory structure
        - total_samples: Total sample count
        - tasks: Dict of task info (use "_root" for root-level data files)

        Args:
            involving_datasets: List of dataset names to include
            datasets_path: Base path for datasets (e.g., "/assets/datasets/")
        """
        datasets_dir = Path(FT_RD_SETTING.file_path) / "datasets"
        dataset_info_path = datasets_dir / "dataset_info.json"

        # Use provided path or get from config
        if datasets_path is None:
            datasets_path = FT_PATHS.datasets

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
            # IMPORTANT: Tell LLM the full path to dataset directory
            dataset_full_path = f"{datasets_path}{name}/"
            info_text += f"- **Dataset path**: `{dataset_full_path}` (each dataset has its own subdirectory)\n"
            info_text += f"- Total samples: {info.get('total_samples', 'N/A')}\n"
            info_text += f"- Size: {info.get('total_size_mb', 'N/A')} MB\n"

            # File tree for understanding directory structure
            if info.get("file_tree"):
                file_tree = info["file_tree"]
                # Truncate if too long
                if len(file_tree) > 1000:
                    file_tree = file_tree[:1000] + "\n..."
                info_text += f"\n**File Structure** (relative to `{dataset_full_path}`):\n```\n{file_tree}\n```\n"

            # Handle unified tasks structure
            tasks = info.get("tasks", {})
            if tasks:
                info_text += "\n**Tasks:**\n"
                for task_name, task_info in tasks.items():
                    # "_root" indicates data files are in root directory
                    display_name = "(root)" if task_name == "_root" else task_name
                    info_text += f"\n#### {display_name}\n"
                    # Show full paths for data files
                    files = task_info.get("files", [])
                    info_text += f"- Files: {files}\n"
                    if files:
                        info_text += f"  - Full path example: `{dataset_full_path}{files[0]}`\n"
                    info_text += f"- Sample count: {task_info.get('sample_count', 'N/A')}\n"
                    if task_info.get("column_stats"):
                        # Show key token stats
                        stats_summary = []
                        for col, stats in task_info["column_stats"].items():
                            if stats.get("p50_tokens", 0) > 0:
                                stats_summary.append(f"{col}: p50={stats['p50_tokens']}, p99={stats['p99_tokens']}")
                        if stats_summary:
                            info_text += f"- Token stats: {'; '.join(stats_summary[:5])}\n"

            # README excerpt
            if info.get("readme"):
                readme = info["readme"]
                if len(readme) > 500:
                    readme = readme[:500] + "..."
                info_text += f"\n**README:**\n{readme}\n"

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
        if prev_task_feedback is not None and prev_task_feedback.source_feedback.get("FTCoderEvaluator", False):
            logger.info("Previous training code passed evaluation, skipping regeneration")
            return {}

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
        config_files = self._generate_llamafactory_config_with_llm(
            base_model=base_model,
            task_info=task_info,
            queried_former_failed_knowledge=queried_former_failed_knowledge,
            prev_feedback=prev_task_feedback,
            workspace=workspace,
        )

        # Return generated config files directly - validation happens in evaluator
        return config_files

    def _generate_llamafactory_config_with_llm(
        self,
        base_model: str,
        task_info: str = "",
        queried_former_failed_knowledge: tuple = None,
        prev_feedback=None,
        workspace=None,
    ) -> dict[str, str]:
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
            deepspeed_path=FT_PATHS.deepspeed,
            data_stats=data_stats,
            has_think_token=self.scen.model_info.get("has_think_token", False),
            force_think_token=FT_RD_SETTING.force_think_token,
        )

        # Call LLM to generate config (multi-turn)
        session = APIBackend().build_chat_session(session_system_prompt=system_prompt)

        # Turn 1: Generate main training config
        train_config_yaml = session.build_chat_completion(
            user_prompt=user_prompt,
            json_mode=False,
            code_block_language="yaml",
            code_block_fallback=False,
        )

        # Validate main config YAML syntax
        yaml.safe_load(train_config_yaml)
        logger.info("Extracted main YAML config successfully")

        # Turn 2: Generate test parameters (test_params.yaml)
        test_params_prompt = T(".prompts:finetune_coder.user_test_params").r(workspace_path=FT_PATHS.workspace)
        test_params_yaml = session.build_chat_completion(
            user_prompt=test_params_prompt,
            json_mode=False,
            code_block_language="yaml",
            code_block_fallback=False,
        )

        # Validate test params YAML syntax
        yaml.safe_load(test_params_yaml)
        logger.info("Extracted test params YAML successfully")

        return {FT_YAML_FILE_NAME: train_config_yaml, FT_TEST_PARAMS_FILE_NAME: test_params_yaml}


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
