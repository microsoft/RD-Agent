import re
from typing import Optional

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.data_science.utils import remove_eda_part
from rdagent.components.coder.finetune.conf import get_clear_ws_cmd, get_ft_env
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.utils.agent.tpl import T


class LLMFinetuneEvaluator(CoSTEEREvaluator):
    """LLM Fine-tuning specific evaluator that uses LLM Docker environment."""

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: Optional[QueriedKnowledge] = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """Evaluate LLM fine-tuning implementation using dedicated LLM environment."""

        # Use LLM-specific environment with appropriate timeout for training
        # For runner, use full timeout instead of debug timeout
        timeout_period = getattr(self.scen, "real_full_timeout", lambda: 3600)()
        env = get_ft_env(
            extra_volumes={FT_RD_SETTING.local_data_path: "/workspace/llm_finetune/data"},
            running_timeout_period=timeout_period,
        )

        # Clean workspace before execution
        stdout = implementation.execute(env=env, entry=get_clear_ws_cmd())

        # Execute LlamaFactory training instead of Python script
        # Check if config.yaml exists in the workspace
        if "config.yaml" not in implementation.file_dict:
            return CoSTEERSingleFeedback(
                execution="No config.yaml found in workspace for LlamaFactory training",
                return_checking="Config file missing",
                code="No valid configuration file",
                final_decision=False,
            )

        # Execute LlamaFactory training
        result = implementation.run(env=env, entry="llamafactory-cli train config.yaml")
        stdout = result.stdout
        execute_ret_code = result.exit_code
        implementation.running_info.running_time = result.running_time

        # Process EDA output if present (for backward compatibility)
        match = re.search(
            r"(.*?)=== Start of EDA part ===(.*)=== End of EDA part ===",
            stdout,
            re.DOTALL,
        )
        eda_output = match.groups()[1] if match else None
        if eda_output is None:
            eda_output = "No EDA output."
        implementation.inject_files(**{"EDA.md": eda_output})
        stdout = remove_eda_part(stdout)

        # Add execution status
        success_msg = "successfully" if execute_ret_code == 0 else "failed"
        stdout += f"The fine-tuning code executed {success_msg}. "
        if eda_output:
            stdout += "The EDA output is removed from the stdout. "

        # Check for model output files (adapter weights, metrics, etc.)
        model_output_files = []
        workspace_path = implementation.workspace_path

        # Look for common LLM fine-tuning output files
        for pattern in [
            "*.safetensors",
            "*.bin",
            "adapter_*",
            "training_*.json",
            "*_metrics.json",
        ]:
            model_output_files.extend(workspace_path.glob(pattern))

        model_check_text = ""
        if model_output_files:
            model_check_text = f"Found model output files: {[f.name for f in model_output_files]}"
        else:
            model_check_text = "No model output files found."

        # Combine all feedback
        full_stdout = f"{stdout}\n\nModel Output Check:\n{model_check_text}"

        return CoSTEERSingleFeedback(
            execution=full_stdout,
            return_checking=model_check_text,
            code=stdout,
            final_decision=execute_ret_code == 0,
        )


class LLMPipelineEvaluator(CoSTEEREvaluator):
    """LLM Pipeline evaluator that uses LLM environment for pipeline tasks."""

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: Optional[QueriedKnowledge] = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """Evaluate LLM pipeline implementation."""

        target_task_information = target_task.get_task_information()

        # Check if task already succeeded
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return CoSTEERSingleFeedback(
                execution="This task has failed too many times, skip implementation.",
                return_checking="This task has failed too many times, skip implementation.",
                code="This task has failed too many times, skip implementation.",
                final_decision=False,
            )

        # Use LLM-specific environment
        env = get_ft_env(
            extra_volumes={FT_RD_SETTING.local_data_path: "/workspace/llm_finetune/data"},
            running_timeout_period=self.scen.real_debug_timeout(),
        )

        # Clean workspace and execute
        implementation.execute(env=env, entry=get_clear_ws_cmd())

        # Run with coverage and strace for debugging
        if hasattr(FT_RD_SETTING, "sample_data_by_LLM") and FT_RD_SETTING.sample_data_by_LLM:
            result = implementation.run(
                env=env,
                entry="strace -e trace=file -f -o trace.log python -m coverage run main.py --debug",
            )
        else:
            result = implementation.run(
                env=env,
                entry="strace -e trace=file -f -o trace.log python -m coverage run main.py",
            )

        stdout = result.stdout
        execute_ret_code = result.exit_code

        # Process output similar to DS pipeline
        if stdout is None:
            stdout = "No output captured from execution."

        return CoSTEERSingleFeedback(
            execution=stdout,
            return_checking=f"Pipeline execution {'succeeded' if execute_ret_code == 0 else 'failed'}",
            code=stdout,
            final_decision=execute_ret_code == 0,
        )
