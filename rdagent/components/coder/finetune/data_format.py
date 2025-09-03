"""
Data Format Converter using CoSTEER architecture for LLM Fine-tuning

This module provides a standardized data format conversion component that can be
easily integrated into RDLoop or used standalone.
"""

from pathlib import Path
from typing import Any, Dict

from rdagent.components.coder.CoSTEER import CoSTEER
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

        # Get file tree and real dataset samples
        file_tree, data_samples = self._get_dataset_info(target_task.dataset)

        if prev_task_feedback is None:
            # First attempt
            user_prompt = T("scenarios.finetune.data_process.prompts:data_format_task.user").r(
                dataset=target_task.dataset,
                runtime_info="Docker environment with mounted data",
                file_tree=file_tree,
                data_samples=data_samples,
            )
        else:
            # Retry with feedback
            user_prompt = T("components.coder.finetune.prompts:data_format_retry").r(
                dataset=target_task.dataset,
                file_tree=file_tree,
                data_samples=data_samples,
                prev_code=workspace.all_codes if workspace else "",
                feedback=prev_task_feedback,
            )

        # Generate code using LLM
        api = APIBackend()
        raw_response = api.build_messages_and_create_chat_completion(
            system_prompt=T("scenarios.finetune.data_process.prompts:data_format_task.system").r(),
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

    def _get_dataset_info(self, dataset: str) -> tuple[str, str]:
        """Get file tree and real dataset samples separately using inherited data science functionality."""

        from rdagent.scenarios.finetune.scen.utils import FinetuneDatasetDescriptor

        try:
            # Use FT_FILE_PATH structure: /path/to/finetune/dataset/<dataset>
            from rdagent.app.finetune.llm.conf import FT_RD_SETTING

            if not FT_RD_SETTING.file_path:
                return "FT_FILE_PATH environment variable not set", "No data samples available"

            dataset_path = Path(FT_RD_SETTING.file_path) / "dataset" / dataset

            if not dataset_path.exists():
                error_msg = f"Dataset {dataset} not found at {dataset_path}"
                return error_msg, error_msg

            # Use the specialized descriptor to get separated info
            descriptor = FinetuneDatasetDescriptor()
            return descriptor.get_separated_info(dataset_path)

        except Exception as e:
            logger.warning(f"Could not generate dataset information: {e}")
            error_msg = f"Error: {str(e)}"
            return error_msg, error_msg

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


class DataFormatCoSTEER(CoSTEER):
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
            with_knowledge=False,
            knowledge_self_gen=False,
            evolving_version=2,  # 使用版本2，与其他CoSTEER实现保持一致
            max_loop=3,
            **kwargs,
        )
