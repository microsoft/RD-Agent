"""
Data Format Converter using CoSTEER architecture for LLM Fine-tuning

This module provides a standardized data format conversion component that can be
easily integrated into RDLoop or used standalone.
"""

from pathlib import Path
from typing import Any, Dict

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiEvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.components.coder.data_science.share.ds_costeer import DSCoSTEER
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class DataFormatTask(CoSTEERTask):
    """Task for data format conversion"""

    def __init__(self, dataset: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.name = f"DataFormat_{dataset}"
        self.description = f"Convert {dataset} to LLaMA-Factory format"

    def get_task_information(self) -> str:
        return f"Convert dataset '{self.dataset}' to LLaMA-Factory compatible format"


class DataFormatEvolvingStrategy(MultiProcessEvolvingStrategy):
    """Evolving strategy for data format conversion"""

    def implement_one_task(
        self,
        target_task: DataFormatTask,
        queried_knowledge=None,
        workspace=None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        """Generate data format conversion code"""

        # Get real dataset samples
        data_samples = self._get_dataset_samples(target_task.dataset)

        if prev_task_feedback is None:
            # First attempt
            user_prompt = T("scenarios.finetune.data_process.prompts:data_format_task_prompt").r(
                dataset=target_task.dataset,
                runtime_info="Docker environment with mounted data",
                data_samples=data_samples,
            )
        else:
            # Retry with feedback
            user_prompt = T("components.coder.finetune.data_format_prompts:data_format_retry").r(
                dataset=target_task.dataset,
                data_samples=data_samples,
                prev_code=workspace.all_codes if workspace else "",
                feedback=prev_task_feedback,
            )

        # Generate code using LLM
        api = APIBackend()
        raw_response = api.build_messages_and_create_chat_completion(
            system_prompt=T("scenarios.finetune.data_process.prompts:data_format_system_prompt").r(),
            user_prompt=user_prompt,
        )

        # Extract Python code using existing utility
        from rdagent.utils.agent.ret import PythonAgentOut

        try:
            code = PythonAgentOut.extract_output(raw_response)
            return {"main.py": code}
        except Exception as e:
            logger.error(f"Failed to extract Python code: {e}")
            raise RuntimeError(f"Failed to generate valid Python code for data format conversion: {e}")

    def _get_dataset_samples(self, dataset: str) -> str:
        """Get comprehensive dataset information including file structure and samples."""
        import os

        from rdagent.scenarios.finetune.scen.utils import build_folder_description

        try:
            # Use FT_FILE_PATH structure: /path/to/finetune/dataset/<dataset>
            ft_file_path = os.environ.get("FT_FILE_PATH")
            if not ft_file_path:
                return "FT_FILE_PATH environment variable not set"

            dataset_path = Path(ft_file_path) / "dataset" / dataset

            if not dataset_path.exists():
                return f"Dataset {dataset} not found at {dataset_path}"

            # Get detailed folder description using unified approach
            folder_description = build_folder_description(dataset)

            # Combine folder structure with sample data information
            result_parts = [
                f"## Dataset: {dataset}",
                "",
                "### File Structure and Content Analysis:",
                folder_description,
                "",
                "### Additional Information:",
                f"Dataset location: {dataset_path}",
                f"This dataset structure analysis is provided to help understand the data format for LLaMA-Factory conversion.",
            ]

            return "\n".join(result_parts)

        except Exception as e:
            logger.warning(f"Could not generate comprehensive dataset information: {e}")
            # Fallback to simple sample extraction for backward compatibility
            return self._get_simple_dataset_samples(dataset)

    def _get_simple_dataset_samples(self, dataset: str) -> str:
        """Fallback method to get simple dataset samples (original implementation)."""
        import json
        import os
        from pathlib import Path

        import pandas as pd

        try:
            # Use FT_FILE_PATH structure consistently
            ft_file_path = os.environ.get("FT_FILE_PATH")
            if not ft_file_path:
                return "FT_FILE_PATH environment variable not set"

            dataset_path = Path(ft_file_path) / "dataset" / dataset

            if not dataset_path.exists():
                return f"Dataset {dataset} not found at {dataset_path}."

            # Find data files in the dataset directory
            data_files = []
            for ext in ["*.json", "*.jsonl", "*.csv", "*.parquet"]:
                data_files.extend(list(dataset_path.glob(ext)))

            if not data_files:
                return f"No supported data files found in {dataset_path}. Supported formats: json, jsonl, csv, parquet"

            # Load sample from the first data file
            data_file = data_files[0]
            sample_data = None

            if data_file.suffix.lower() == ".json":
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        sample_data = data[0]
                    elif isinstance(data, dict):
                        sample_data = data

            elif data_file.suffix.lower() == ".jsonl":
                with open(data_file, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        sample_data = json.loads(first_line)

            elif data_file.suffix.lower() == ".csv":
                df = pd.read_csv(data_file)
                if len(df) > 0:
                    sample_data = df.iloc[0].to_dict()

            elif data_file.suffix.lower() == ".parquet":
                df = pd.read_parquet(data_file)
                if len(df) > 0:
                    sample_data = df.iloc[0].to_dict()

            if sample_data is not None:
                return json.dumps(sample_data, ensure_ascii=False, indent=2)
            else:
                return f"Could not extract sample from {data_file.name}"

        except Exception as e:
            logger.warning(f"Could not load dataset samples: {e}")
            return f"Dataset: {dataset}\nError loading samples: {str(e)}"

    def assign_code_list_to_evo(self, code_list: list[dict[str, str]], evo):
        """Assign generated code to evolving item"""
        from rdagent.core.experiment import FBWorkspace

        for index, code_dict in enumerate(code_list):
            if code_dict is not None:
                # 确保 sub_workspace_list 有足够的空间
                if evo.sub_workspace_list[index] is None:
                    evo.sub_workspace_list[index] = FBWorkspace()
                evo.sub_workspace_list[index].inject_files(**code_dict)

        return evo


class DataFormatEvaluator(CoSTEEREvaluator):
    """Evaluator for data format conversion results"""

    def evaluate(
        self,
        target_task: Task,
        implementation,
        gt_implementation,
        queried_knowledge=None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """Evaluate the data format conversion implementation"""

        # Check if implementation exists
        if implementation is None:
            return CoSTEERSingleFeedback(
                execution="No implementation provided",
                return_checking="Implementation workspace is None",
                code="No code to evaluate",
                final_decision=False,
            )

        # For data format conversion, we mainly check if the code was generated
        # The actual file existence check will happen during execution
        if not hasattr(implementation, "file_dict") or not implementation.file_dict:
            return CoSTEERSingleFeedback(
                execution="Empty implementation",
                return_checking="No code files found in workspace",
                code="Implementation is empty",
                final_decision=False,
            )

        # Check if main.py exists
        if "main.py" not in implementation.file_dict:
            return CoSTEERSingleFeedback(
                execution="Missing main.py",
                return_checking="Required main.py file not found",
                code="Implementation missing main execution file",
                final_decision=False,
            )

        # Basic code quality check
        main_code = implementation.file_dict["main.py"]
        if len(main_code.strip()) < 50:  # Very basic check
            return CoSTEERSingleFeedback(
                execution="Code too short",
                return_checking="Generated code appears to be too minimal",
                code="Implementation seems incomplete",
                final_decision=False,
            )

        return CoSTEERSingleFeedback(
            execution="Code generation successful",
            return_checking="Main script generated with reasonable content",
            code="Implementation looks good",
            final_decision=True,
        )


class DataFormatCoSTEER(DSCoSTEER):
    """CoSTEER implementation for data format conversion"""

    def __init__(self, scen: Scenario, *args, **kwargs):
        settings = CoSTEERSettings(max_loop=3)
        eva = CoSTEERMultiEvaluator(DataFormatEvaluator(scen=scen), scen=scen)
        es = DataFormatEvolvingStrategy(scen=scen, settings=settings)

        super().__init__(
            settings=settings,
            eva=eva,
            es=es,
            scen=scen,
            evolving_version=2,  # 使用版本2，与其他CoSTEER实现保持一致
            max_loop=3,
            **kwargs,
        )
