from typing import Optional

import yaml

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
from rdagent.scenarios.finetune.train.benchmark import run_benchmark


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
            return CoSTEERSingleFeedback(
                execution=f"No {FT_YAML_FILE_NAME} found in workspace",
                return_checking="Config file missing",
                code="No valid configuration file",
                final_decision=False,
            )

        config = yaml.safe_load(implementation.file_dict.get(FT_YAML_FILE_NAME, ""))
        test_config = config.copy()
        test_config.update(
            {
                "num_train_epochs": 1,
                "max_steps": 20,
                "save_steps": 20,
                "logging_steps": 10,
                "warmup_steps": 0,
                "overwrite_output_dir": True,
                "report_to": "none",  # Disable all reporting (tensorboard, wandb, etc.)
                "do_eval": False,  # Disable evaluation in micro-batch test (insufficient samples for val split)
                "eval_strategy": "no",  # Explicitly disable evaluation
                "load_best_model_at_end": False,  # Cannot load best model without evaluation
            }
        )
        # Run micro-batch training
        implementation.inject_files(**{FT_YAML_FILE_NAME: yaml.dump(test_config, default_flow_style=False)})

        # Execute LlamaFactory training
        result = implementation.run(env=env, entry=f"llamafactory-cli train {FT_YAML_FILE_NAME}")
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

        # Use open-compass to evaluate the model on benchmark
        benchmark_result = run_benchmark(
            workspace_path=str(workspace_path),
            model_path=output_path,
            model_name=target_task.base_model,
            benchmark_name=target_task.benchmark,
        )

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
