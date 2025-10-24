import re
from typing import Optional

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.finetune.conf import get_clear_ws_cmd, get_ft_env
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T


class FTRunnerEvaluator(CoSTEEREvaluator):
    """LLM Fine-tuning specific evaluator that uses LLM Docker environment."""

    def _analyze_training_log(self, stdout: str) -> dict:
        """
        Analyze training log to extract key metrics and detect critical errors.

        Returns:
            dict: Analysis results including loss values, error detection, and training status
        """
        analysis = {
            "has_loss": False,
            "loss_values": [],
            "train_loss": None,
            "loss_trend": "unknown",
            "critical_errors": [],
            "training_completed": False,
        }

        # Check for critical errors
        error_patterns = [
            (r"loss['\"]?\s*:\s*nan|loss is nan", "Loss became NaN - training diverged"),
            (r"cuda out of memory|CUDA error: out of memory", "CUDA out of memory"),
            (r"killed|Killed", "Process killed (likely OOM)"),
            (r"RuntimeError|ValueError|TypeError", "Runtime error during training"),
        ]

        for pattern, error_msg in error_patterns:
            if re.search(pattern, stdout, re.IGNORECASE):
                analysis["critical_errors"].append(error_msg)
                logger.warning(f"Critical error detected in training log: {error_msg}")

        # Extract loss values from training log
        # LlamaFactory format: {'loss': 1.5268, 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}
        loss_matches = re.findall(r"['\"]loss['\"]\s*:\s*([\d.]+)", stdout)
        if loss_matches:
            analysis["has_loss"] = True
            analysis["loss_values"] = [float(l) for l in loss_matches]

        # Extract final train_loss from summary
        # Format: {'train_runtime': ..., 'train_loss': 1.5268092155456543, 'epoch': ...}
        train_loss_match = re.search(r"['\"]train_loss['\"]\s*:\s*([\d.]+)", stdout)
        if train_loss_match:
            analysis["train_loss"] = float(train_loss_match.group(1))

        # Analyze loss trend
        if len(analysis["loss_values"]) >= 2:
            first_loss = analysis["loss_values"][0]
            last_loss = analysis["loss_values"][-1]
            if last_loss < first_loss * 0.95:
                analysis["loss_trend"] = "decreasing"
            elif last_loss > first_loss * 1.05:
                analysis["loss_trend"] = "increasing"
            else:
                analysis["loss_trend"] = "stable"
        elif len(analysis["loss_values"]) == 1:
            analysis["loss_trend"] = "single_value"

        # Check if training completed successfully
        if re.search(r"Training completed|Saving model checkpoint", stdout):
            analysis["training_completed"] = True

        return analysis

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

        # Analyze training log for detailed insights
        log_analysis = self._analyze_training_log(stdout)

        # Build execution feedback message
        execution_parts = []
        success_msg = "successfully" if execute_ret_code == 0 else "failed"
        execution_parts.append(f"Training execution {success_msg}.")

        # Add loss information
        if log_analysis["has_loss"]:
            if log_analysis["train_loss"] is not None:
                execution_parts.append(f"Final train loss: {log_analysis['train_loss']:.4f}")
            if log_analysis["loss_values"]:
                execution_parts.append(
                    f"Loss trend: {log_analysis['loss_trend']} "
                    f"(first: {log_analysis['loss_values'][0]:.4f}, "
                    f"last: {log_analysis['loss_values'][-1]:.4f})"
                )
        else:
            execution_parts.append("Warning: No loss values found in training log.")

        # Add critical error warnings
        if log_analysis["critical_errors"]:
            execution_parts.append(f"Critical errors detected: {', '.join(log_analysis['critical_errors'])}")

        # Add training completion status
        if log_analysis["training_completed"]:
            execution_parts.append("Training completed successfully.")
        elif execute_ret_code == 0:
            execution_parts.append("Warning: Training finished but completion message not found.")

        execution_summary = " ".join(execution_parts)

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

        # Determine final decision based on multiple factors
        final_decision = (
            execute_ret_code == 0 and len(log_analysis["critical_errors"]) == 0 and log_analysis["training_completed"]
        )

        # Build comprehensive feedback
        feedback_parts = [
            execution_summary,
            f"\nModel Output Check: {model_check_text}",
        ]

        # Add detailed log analysis for debugging (truncated)
        if not final_decision:
            feedback_parts.append(f"\nTraining log excerpt (last 1000 chars):\n{stdout[-1000:]}")

        return CoSTEERSingleFeedback(
            execution="\n".join(feedback_parts),
            return_checking=model_check_text,
            code=execution_summary,
            final_decision=final_decision,
        )
