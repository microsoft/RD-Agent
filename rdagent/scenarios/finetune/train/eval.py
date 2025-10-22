from typing import Optional

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
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
            running_timeout_period=timeout_period,
        )

        # Clean workspace before execution
        stdout = implementation.execute(env=env, entry=get_clear_ws_cmd())

        # Execute LlamaFactory training instead of Python script
        # Check if train.yaml exists in the workspace
        if "train.yaml" not in implementation.file_dict:
            return CoSTEERSingleFeedback(
                execution="No train.yaml found in workspace for LlamaFactory training",
                return_checking="Config file missing",
                code="No valid configuration file",
                final_decision=False,
            )

        # Execute LlamaFactory training
        result = implementation.run(env=env, entry="llamafactory-cli train train.yaml")
        stdout = result.stdout
        execute_ret_code = result.exit_code
        implementation.running_info.running_time = result.running_time

        # Add execution status
        success_msg = "successfully" if execute_ret_code == 0 else "failed"
        stdout += f"The fine-tuning code executed {success_msg}. "

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
