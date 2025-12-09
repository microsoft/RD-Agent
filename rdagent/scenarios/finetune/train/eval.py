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
from rdagent.components.coder.finetune.unified_validator import LLMConfigValidator
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.train.benchmark import run_benchmark
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry


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
        env = get_ft_env(operation="full_training")

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
            implementation.feedback = fb
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
        model_output_files = (
            list(output_path.glob("*.safetensors"))
            + list(output_path.glob("*.bin"))
            + list(output_path.glob("adapter_*"))
            if output_path.exists()
            else []
        )

        # Early return if training failed
        if not training_success or len(model_output_files) == 0:
            if not output_path.exists():
                error_msg = f"Output directory not found (exit_code={result.exit_code})"
            elif not training_success:
                error_msg = f"Training failed (exit_code={result.exit_code})"
            else:
                error_msg = "No model output files generated"
            return self._generate_llm_feedback(
                target_task=target_task,
                implementation=implementation,
                raw_stdout=raw_stdout,
                exit_code=result.exit_code,
                training_success=False,
                error_msg=error_msg,
            )

        # Extract loss history from training output
        loss_history = extract_loss_history(output_path)

        # Use open-compass to evaluate the model on benchmark (only if training succeeded)
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

        # Final decision: training succeeded AND model files exist AND benchmark ran
        final_decision = training_success and len(model_output_files) > 0 and benchmark_result is not None

        # Call LLM for feedback analysis (both success and failure cases)
        if final_decision:
            # Success: analyze benchmark results and training metrics
            return self._generate_llm_feedback(
                target_task=target_task,
                implementation=implementation,
                raw_stdout=raw_stdout,
                exit_code=result.exit_code,
                training_success=True,
                benchmark_result=benchmark_result,
                loss_history=loss_history,
            )
        else:
            # Failure: analyze error cause
            error_msg = f"exit_code={result.exit_code}"
            if not training_success:
                error_msg = f"Training failed: {error_msg}"
            elif len(model_output_files) == 0:
                error_msg = "No model output files generated"
            elif benchmark_result is None:
                error_msg = "Benchmark evaluation failed"
            return self._generate_llm_feedback(
                target_task=target_task,
                implementation=implementation,
                raw_stdout=raw_stdout,
                exit_code=result.exit_code,
                training_success=False,
                error_msg=error_msg,
            )

    def _generate_llm_feedback(
        self,
        target_task: FTTask,
        implementation: FBWorkspace,
        raw_stdout: str,
        exit_code: int,
        training_success: bool,
        benchmark_result: Optional[Dict] = None,
        loss_history: Optional[List[Dict]] = None,
        error_msg: Optional[str] = None,
    ) -> CoSTEERSingleFeedback:
        """Generate LLM-based feedback for runner evaluation."""
        version = "runner_eval" if training_success else "runner_eval_error"

        # Parse execution log to extract structured info (reuse unified_validator's method)
        # Reduces ~36k tokens to ~500 tokens by extracting: status, errors, metrics, warnings
        parsed_stdout = LLMConfigValidator()._parse_execution_log(raw_stdout, exit_code)

        # Build loss summary instead of raw history (saves tokens, provides key insights)
        loss_summary = {}
        if loss_history:
            losses = [e["loss"] for e in loss_history]
            min_idx = losses.index(min(losses))
            loss_summary = {
                "logged_entries": len(loss_history),
                "final_step": loss_history[-1].get("step"),
                "initial_loss": round(loss_history[0]["loss"], 4),
                "final_loss": round(loss_history[-1]["loss"], 4),
                "min_loss": round(min(losses), 4),
                "min_loss_at_entry": min_idx + 1,  # 1-indexed
                "loss_trend": "rising_late" if losses[-1] > min(losses) * 1.1 else "stable",
            }

        system_prompt = T(f"rdagent.components.coder.finetune.prompts:{version}.system").r()
        user_prompt = T(f"rdagent.components.coder.finetune.prompts:{version}.user").r(
            task_desc=target_task.get_task_information(),
            config_yaml=implementation.file_dict.get(FT_YAML_FILE_NAME, ""),
            stdout=parsed_stdout,  # Structured JSON instead of raw truncated log
            benchmark_result=json.dumps(benchmark_result, indent=2) if benchmark_result else "N/A",
            loss_summary=json.dumps(loss_summary, indent=2) if loss_summary else "N/A",
            error_msg=error_msg or "",
        )

        feedback = build_cls_from_json_with_retry(
            CoSTEERSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=CoSTEERSingleFeedback.val_and_update_init_dict,
        )
        feedback.raw_execution = raw_stdout
        implementation.feedback = feedback
        logger.log_object(feedback, tag="evaluator_feedback.FTRunnerEvaluator")
        return feedback
