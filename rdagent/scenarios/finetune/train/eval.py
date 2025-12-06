import json
from typing import Any, Dict, List, Optional

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.finetune.conf import (
    FT_YAML_FILE_NAME,
    get_clear_ws_cmd,
    get_ft_env,
)
from rdagent.components.coder.finetune.exp import FTTask
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.train.benchmark import run_benchmark


def extract_loss_history(output_path) -> List[Dict[str, Any]]:
    """
    Extract training loss history from LlamaFactory's trainer_state.json.

    Args:
        output_path: Path to the training output directory

    Returns:
        List of loss entries, each containing step, loss, and epoch
    """
    trainer_state_path = output_path / "trainer_state.json"
    loss_history = []

    if not trainer_state_path.exists():
        logger.warning(f"trainer_state.json not found at {trainer_state_path}")
        return loss_history

    try:
        with open(trainer_state_path) as f:
            trainer_state = json.load(f)

        log_history = trainer_state.get("log_history", [])
        for entry in log_history:
            if "loss" in entry:
                loss_history.append(
                    {
                        "step": entry.get("step"),
                        "loss": entry.get("loss"),
                        "epoch": entry.get("epoch"),
                    }
                )

        logger.info(f"Extracted {len(loss_history)} loss entries from trainer_state.json")

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to parse trainer_state.json: {e}")

    return loss_history


class FTRunnerEvaluator(CoSTEEREvaluator):
    """LLM Fine-tuning specific evaluator that uses LLM Docker environment."""

    def evaluate(
        self,
        target_task: FTTask,
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

        # Check if FT_YAML_FILE_NAME exists
        if FT_YAML_FILE_NAME not in implementation.file_dict:
            fb = CoSTEERSingleFeedback(
                execution=f"No {FT_YAML_FILE_NAME} found in workspace",
                return_checking="Config file missing",
                code="No valid configuration file",
                final_decision=False,
            )
            logger.log_object(fb, tag="evaluator_feedback.FTRunnerEvaluator")
            return fb

        # Execute LlamaFactory training
        result = implementation.run(
            env=env,
            entry=f"llamafactory-cli train {FT_YAML_FILE_NAME}",
        )
        implementation.running_info.running_time = result.running_time
        raw_stdout = result.stdout or ""
        # NOTE: Docker execution is logged by FTWorkspace.run() automatically

        # Simple success check: exit code
        training_success = result.exit_code == 0

        # Check for model output files
        workspace_path = implementation.workspace_path
        output_path = workspace_path / "output"
        if not output_path.exists():
            fb = CoSTEERSingleFeedback(
                execution=f"Output directory not found (exit_code={result.exit_code})",
                return_checking="Training failed - no output generated",
                code="Check training logs for errors",
                final_decision=False,
            )
            fb.raw_execution = raw_stdout
            logger.log_object(fb, tag="evaluator_feedback.FTRunnerEvaluator")
            return fb
        model_output_files = []
        for pattern in ["*.safetensors", "*.bin", "adapter_*"]:
            model_output_files.extend(output_path.glob(pattern))

        # Extract loss history from training output
        loss_history = extract_loss_history(output_path)

        # Use open-compass to evaluate the model on benchmark
        benchmark_result = run_benchmark(
            workspace_path=str(workspace_path),
            model_path=output_path,
            model_name=target_task.base_model,
            benchmark_name=target_task.benchmark,
        )

        # Build comprehensive result with training metrics and benchmark results
        implementation.running_info.result = {
            "benchmark": benchmark_result,  # Contains accuracy_summary and error_samples
            "training_metrics": {
                "loss_history": loss_history,
                "final_loss": loss_history[-1]["loss"] if loss_history else None,
                "initial_loss": loss_history[0]["loss"] if loss_history else None,
            },
        }

        # Final decision: training succeeded AND model files exist
        final_decision = training_success and len(model_output_files) > 0 and benchmark_result is not None

        # Build minimal feedback
        execution_msg = f"Training {'succeeded' if training_success else 'failed'} (exit_code={result.exit_code})"
        if model_output_files:
            model_msg = f"Found {len(model_output_files)} model output files"
        else:
            model_msg = "No model output files found"

        if benchmark_result:
            accuracy_summary = benchmark_result.get("accuracy_summary", [])
            model_msg += f"; Benchmark result: {accuracy_summary}"

        feedback_msg = f"{execution_msg}. {model_msg}."

        fb = CoSTEERSingleFeedback(
            execution=feedback_msg,
            return_checking=model_msg,
            code=execution_msg,
            final_decision=final_decision,
        )
        fb.raw_execution = raw_stdout
        logger.log_object(fb, tag="evaluator_feedback.FTRunnerEvaluator")
        return fb
