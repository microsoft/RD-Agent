from typing import Optional

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.finetune.conf import get_clear_ws_cmd, get_ft_env
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task


class FTRunnerEvaluator(CoSTEEREvaluator):
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
        timeout_period = getattr(self.scen, "real_full_timeout", lambda: 3600)()

        env = get_ft_env(running_timeout_period=timeout_period)

        # Clean workspace before execution
        implementation.execute(env=env, entry=get_clear_ws_cmd())

        # Check if train.yaml exists
        if "train.yaml" not in implementation.file_dict:
            return CoSTEERSingleFeedback(
                execution="No train.yaml found in workspace",
                return_checking="Config file missing",
                code="No valid configuration file",
                final_decision=False,
            )

        # Execute LlamaFactory training
        result = implementation.run(env=env, entry="llamafactory-cli train train.yaml")
        implementation.running_info.running_time = result.running_time

        # Simple success check: exit code
        training_success = result.exit_code == 0

        # Check for model output files
        workspace_path = implementation.workspace_path
        output_path = workspace_path / "output"
        if not output_path.exists():
            return CoSTEERSingleFeedback(
                execution="Output directory not found",
                return_checking="Output directory not found",
                code="Output directory not found",
                final_decision=False,
            )
        model_output_files = []
        for pattern in ["*.safetensors", "*.bin", "adapter_*"]:
            model_output_files.extend(output_path.glob(pattern))

        # Final decision: training succeeded AND model files exist
        final_decision = training_success and len(model_output_files) > 0

        # Build minimal feedback
        execution_msg = f"Training {'succeeded' if training_success else 'failed'} (exit_code={result.exit_code})"
        if model_output_files:
            model_msg = f"Found {len(model_output_files)} model output files"
        else:
            model_msg = "No model output files found"

        feedback_msg = f"{execution_msg}. {model_msg}."

        return CoSTEERSingleFeedback(
            execution=feedback_msg,
            return_checking=model_msg,
            code=execution_msg,
            final_decision=final_decision,
        )
