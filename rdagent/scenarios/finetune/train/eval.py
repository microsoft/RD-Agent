import json
from typing import Any, Dict, List, Optional

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.finetune.conf import (
    FT_DATA_FILE_NAME,
    FT_DATA_SCRIPT_NAME,
    FT_YAML_FILE_NAME,
    clear_workspace,
    get_data_processing_cache_key,
    get_data_processing_env,
    get_ft_env,
    get_workspace_prefix,
    inject_data_stats,
)
from rdagent.components.coder.finetune.exp import FTTask
from rdagent.components.coder.finetune.unified_validator import LLMConfigValidator
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.benchmark import get_benchmark_ranges, run_benchmark
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry


def extract_loss_history(output_path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract training and evaluation loss history from LlamaFactory's trainer_state.json.

    Args:
        output_path: Path to the training output directory

    Returns:
        Dict with 'train' and 'eval' keys, each containing a list of loss entries.
    """
    trainer_state_path = output_path / "trainer_state.json"
    result = {"train": [], "eval": []}

    if not trainer_state_path.exists():
        logger.warning(f"trainer_state.json not found at {trainer_state_path}")
        return result

    try:
        with open(trainer_state_path) as f:
            trainer_state = json.load(f)

        log_history = trainer_state.get("log_history", [])
        for entry in log_history:
            if "loss" in entry:
                result["train"].append(
                    {
                        "step": entry.get("step"),
                        "epoch": entry.get("epoch"),
                        "loss": entry.get("loss"),
                    }
                )
            if "eval_loss" in entry:
                result["eval"].append(
                    {
                        "step": entry.get("step"),
                        "epoch": entry.get("epoch"),
                        "eval_loss": entry.get("eval_loss"),
                    }
                )

        logger.info(f"Extracted {len(result['train'])} train + {len(result['eval'])} eval entries")

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to parse trainer_state.json: {e}")

    return result


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
        """Evaluate LLM fine-tuning implementation using dedicated LLM environment.

        This evaluator performs three stages:
        0. Clean workspace (remove old training outputs)
        1. Full data processing (without --debug flag) to generate complete data.json
        2. Full training with the complete dataset
        """

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

        # Use LLM-specific environment with appropriate timeout for training
        env = get_ft_env(operation="full_training")

        # ========== Stage 0: Clean Workspace ==========
        # Clean old training outputs before data processing and training
        clear_workspace(implementation, env)

        # ========== Stage 1: Full Data Processing ==========
        # Execute data processing WITHOUT --debug flag to generate complete data.json
        data_result = self._run_full_data_processing(implementation)
        data_stdout = data_result.stdout or ""

        if data_result.exit_code != 0:
            # Data processing failed, return feedback to enter next loop
            logger.error(f"Full data processing failed with exit_code={data_result.exit_code}")
            return self._generate_llm_feedback(
                target_task=target_task,
                implementation=implementation,
                raw_stdout=data_stdout,
                exit_code=data_result.exit_code,
                model_files_exist=False,
                benchmark_result=None,
                loss_history=None,
                failed_stage="data_processing",
            )

        logger.info("Full data processing completed successfully")

        # Update data_stats.json with full dataset statistics
        # This ensures feedback sees the correct sample count, not debug mode count
        data_json_path = implementation.workspace_path / FT_DATA_FILE_NAME
        if data_json_path.exists():
            with open(data_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                inject_data_stats(implementation, data, data_stdout)

        # ========== Stage 2: Full Training ==========

        # Execute LlamaFactory training
        train_result = implementation.run(
            env=env,
            entry=f"llamafactory-cli train {FT_YAML_FILE_NAME}",
        )
        # Combine data processing and training stdout for comprehensive feedback
        combined_stdout = (
            f"=== DATA PROCESSING OUTPUT ===\n{data_stdout}\n\n=== TRAINING OUTPUT ===\n{train_result.stdout or ''}"
        )
        implementation.running_info.running_time = train_result.running_time
        # NOTE: Docker execution is logged by FTWorkspace.run() automatically

        # Simple success check: exit code
        training_success = train_result.exit_code == 0

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
            return self._generate_llm_feedback(
                target_task=target_task,
                implementation=implementation,
                raw_stdout=combined_stdout,
                exit_code=train_result.exit_code,
                model_files_exist=len(model_output_files) > 0,
                benchmark_result=None,
                loss_history=None,
                failed_stage="training",
            )

        # Extract loss history from training output
        loss_history = extract_loss_history(output_path)

        val_range, test_range = get_benchmark_ranges()

        # Validation set - used for SOTA judgment, visible to agent
        validation_result = run_benchmark(
            workspace_path=str(workspace_path),
            model_path=output_path,
            model_name=target_task.base_model,
            benchmark_name=target_task.benchmark,
            gpu_count=self.scen.gpu_count,
            test_range=val_range,
            result_subdir="validation",
        )

        # Test set - only for frontend display, not visible to agent
        test_result = run_benchmark(
            workspace_path=str(workspace_path),
            model_path=output_path,
            model_name=target_task.base_model,
            benchmark_name=target_task.benchmark,
            gpu_count=self.scen.gpu_count,
            test_range=test_range,
            result_subdir="test",
        )

        # Build comprehensive result with training metrics and benchmark results
        # Note: "benchmark" is for agent (SOTA judgment), "benchmark_test" is for frontend only
        train_history = loss_history.get("train", []) if loss_history else []
        implementation.running_info.result = {
            "benchmark": validation_result,  # Agent visible - used for SOTA judgment
            "benchmark_test": test_result,  # Agent invisible - frontend display only
            "training_metrics": {
                "loss_history": loss_history,
                "final_loss": train_history[-1]["loss"] if train_history else None,
                "initial_loss": train_history[0]["loss"] if train_history else None,
            },
        }
        benchmark_result = validation_result  # For backward compatibility with feedback

        # Call LLM for feedback analysis - LLM will determine final_decision
        return self._generate_llm_feedback(
            target_task=target_task,
            implementation=implementation,
            raw_stdout=combined_stdout,
            exit_code=train_result.exit_code,
            model_files_exist=len(model_output_files) > 0,
            benchmark_result=benchmark_result,
            loss_history=loss_history,
        )

    def _generate_llm_feedback(
        self,
        target_task: FTTask,
        implementation: FBWorkspace,
        raw_stdout: str,
        exit_code: int,
        model_files_exist: bool,
        benchmark_result: Optional[Dict] = None,
        loss_history: Optional[Dict[str, List[Dict]]] = None,
        failed_stage: Optional[str] = None,
    ) -> CoSTEERSingleFeedback:
        """Generate LLM-based feedback for runner evaluation.

        LLM will determine final_decision based on all provided information.

        Args:
            failed_stage: Which stage failed - "data_processing" or "training"
        """
        # Parse execution log to extract structured info (reuse unified_validator's method)
        # Reduces ~36k tokens to ~500 tokens by extracting: status, errors, metrics, warnings
        parsed_stdout = LLMConfigValidator()._parse_execution_log(raw_stdout, exit_code, failed_stage)

        # Get timeout config for the failed stage
        timeout_seconds = None
        if failed_stage == "data_processing":
            timeout_seconds = FT_RD_SETTING.data_processing_timeout
        elif failed_stage == "training":
            timeout_seconds = FT_RD_SETTING.full_timeout

        # Pass loss_history directly (simpler and preserves full information)
        # Sample train entries if too many to avoid token bloat
        if loss_history and len(loss_history.get("train", [])) > 60:
            loss_history["train"] = loss_history["train"][:30] + loss_history["train"][-30:]

        system_prompt = T("rdagent.components.coder.finetune.prompts:runner_eval.system").r()
        user_prompt = T("rdagent.components.coder.finetune.prompts:runner_eval.user").r(
            task_desc=target_task.get_task_information(),
            config_yaml=implementation.file_dict.get(FT_YAML_FILE_NAME, ""),
            exit_code=exit_code,
            model_files_status="Found" if model_files_exist else "Not found",
            stdout=parsed_stdout,  # Structured JSON instead of raw truncated log
            benchmark_result=(
                json.dumps(benchmark_result, indent=2) if benchmark_result else "N/A (not executed or failed)"
            ),
            loss_history=(
                json.dumps(loss_history, indent=2)
                if (loss_history and (loss_history.get("train") or loss_history.get("eval")))
                else "N/A"
            ),
            failed_stage=failed_stage,
            timeout_seconds=timeout_seconds,
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

    def _run_full_data_processing(self, implementation: FBWorkspace):
        """Execute full data processing (without --debug flag) to generate complete data.json.

        This is called at the beginning of the running stage to regenerate data.json
        with all samples instead of the debug subset created during coding stage.

        Args:
            implementation: The workspace containing process_data.py

        Returns:
            EnvResult with exit_code, stdout, etc.
        """
        # Get data processing environment with LLM API access
        env, env_vars = get_data_processing_env()
        ws_prefix = get_workspace_prefix(env)

        logger.info("Starting full data processing (without --debug flag)")

        # Execute WITHOUT --debug flag to generate all samples
        result = implementation.run(
            env=env,
            entry=f"python {ws_prefix}/{FT_DATA_SCRIPT_NAME}",  # No --debug flag
            env_vars=env_vars,
            cache_key_extra_func=get_data_processing_cache_key,
            cache_files_to_extract=[FT_DATA_FILE_NAME],
        )

        return result
