"""
LLM Fine-tuning Evaluation Components

Provides simplified evaluation: parameter filtering + micro-batch testing.
No redundant LLM feedback generation - test results speak for themselves.
"""

import json
from pathlib import Path
from typing import Optional

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.finetune.conf import (
    FT_DATA_FILE_NAME,
    FT_DATA_SCRIPT_NAME,
    FT_YAML_FILE_NAME,
    get_data_processing_env,
    get_ft_env,
)
from rdagent.components.coder.finetune.unified_validator import LLMConfigValidator
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

DIRNAME = Path(__file__).absolute().resolve().parent


class FTDataEvaluator(CoSTEEREvaluator):
    """Evaluator for data processing results.

    This evaluator:
    1. Checks if task_type requires data processing
    2. Executes the process_data.py script in Docker
    3. Validates the output data.json file
    4. Generates dataset_info.json for LlamaFactory
    """

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: Optional[QueriedKnowledge] = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """Evaluate data processing implementation."""

        # 1. Check task_type - skip if train-only
        task_type = getattr(target_task, "task_type", "train")
        if task_type == "train":
            logger.info("Task type is 'train', skipping data evaluation")
            return CoSTEERSingleFeedback(
                execution="Skipped (train-only task)",
                return_checking="",
                code="",
                final_decision=True,
            )

        # 2. Check if data.json already exists and is valid (skip execution)
        data_json_path = implementation.workspace_path / FT_DATA_FILE_NAME
        if data_json_path.exists():
            validation_result = self._validate_data_json(data_json_path)
            if validation_result["valid"]:
                logger.info("Valid data.json already exists, skipping execution")
                # Update dataset_info.json
                self._update_dataset_info(implementation, validation_result["sample_count"])
                return CoSTEERSingleFeedback(
                    execution=f"Data already exists: {validation_result['sample_count']} samples",
                    return_checking="data.json valid",
                    code="",
                    final_decision=True,
                )

        # 3. Check if process_data.py exists
        script_code = implementation.file_dict.get(FT_DATA_SCRIPT_NAME, "")
        if not script_code:
            return CoSTEERSingleFeedback(
                execution=f"No {FT_DATA_SCRIPT_NAME} found",
                return_checking="Data processing script missing",
                code="",
                final_decision=False,
            )

        # 4. Execute the script in Docker
        env, env_vars = get_data_processing_env(
            running_timeout_period=3600,  # 1 hour timeout for data processing
        )

        try:
            # Prepare workspace and inject files
            implementation.prepare()
            implementation.inject_files(**implementation.file_dict)

            # Run with LLM API environment variables
            result = env.run(
                entry=f"python /workspace/{FT_DATA_SCRIPT_NAME}",
                local_path=str(implementation.workspace_path),
                env=env_vars,
            )
            execution_output = result.stdout if hasattr(result, "stdout") else str(result)
            exit_code = result.exit_code if hasattr(result, "exit_code") else -1
        except Exception as e:
            logger.error(f"Failed to execute data processing script: {e}")
            return CoSTEERSingleFeedback(
                execution=f"Script execution failed: {e}",
                return_checking="Execution error",
                code="",
                final_decision=False,
            )

        # 5. Check if data.json was generated
        if not data_json_path.exists():
            return CoSTEERSingleFeedback(
                execution=f"Script executed (exit_code={exit_code}) but {FT_DATA_FILE_NAME} not found.\n"
                f"Output: {execution_output[:2000] if execution_output else 'No output'}",
                return_checking=f"{FT_DATA_FILE_NAME} not generated",
                code="",
                final_decision=False,
            )

        # 6. Validate data.json format
        validation_result = self._validate_data_json(data_json_path)
        if not validation_result["valid"]:
            return CoSTEERSingleFeedback(
                execution=f"Script executed successfully but data validation failed: {validation_result['error']}",
                return_checking=validation_result["error"],
                code="",
                final_decision=False,
            )

        # 7. Update dataset_info.json for LlamaFactory
        self._update_dataset_info(implementation, validation_result["sample_count"])

        logger.info(f"Data processing successful: {validation_result['sample_count']} samples")
        return CoSTEERSingleFeedback(
            execution=f"Successfully processed {validation_result['sample_count']} samples",
            return_checking=f"{FT_DATA_FILE_NAME} valid with {validation_result['sample_count']} samples",
            code="",
            final_decision=True,
        )

    def _validate_data_json(self, data_json_path: Path) -> dict:
        """Validate data.json file format and content."""
        try:
            with open(data_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Must be a non-empty list
            if not isinstance(data, list):
                return {"valid": False, "error": "data.json must be a JSON array", "sample_count": 0}

            if len(data) == 0:
                return {"valid": False, "error": "data.json is empty", "sample_count": 0}

            # Check required fields in samples
            required_fields = ["instruction", "output"]
            for i, sample in enumerate(data[:10]):  # Check first 10 samples
                if not isinstance(sample, dict):
                    return {"valid": False, "error": f"Sample {i} is not a dict", "sample_count": 0}

                missing = [f for f in required_fields if f not in sample]
                if missing:
                    return {"valid": False, "error": f"Sample {i} missing fields: {missing}", "sample_count": 0}

                # Check for empty required fields
                for field in required_fields:
                    if not sample.get(field):
                        return {
                            "valid": False,
                            "error": f"Sample {i} has empty '{field}' field",
                            "sample_count": 0,
                        }

            return {"valid": True, "error": None, "sample_count": len(data)}

        except json.JSONDecodeError as e:
            return {"valid": False, "error": f"Invalid JSON: {e}", "sample_count": 0}
        except Exception as e:
            return {"valid": False, "error": f"Error reading file: {e}", "sample_count": 0}

    def _update_dataset_info(self, implementation: FBWorkspace, sample_count: int):
        """Generate dataset_info.json for LlamaFactory to use the processed data.

        Note: LlamaFactory's columns mapping uses internal names (prompt, query, response)
        that map to the actual column names in the data file (instruction, input, output).
        See: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/parser.py
        """
        dataset_info = {
            "processed_data": {
                "file_name": FT_DATA_FILE_NAME,
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            }
        }

        try:
            implementation.inject_files(**{"dataset_info.json": json.dumps(dataset_info, indent=2)})
            logger.info(f"Updated dataset_info.json with processed_data ({sample_count} samples)")
        except Exception as e:
            logger.warning(f"Failed to update dataset_info.json: {e}")


class FTCoderEvaluator(CoSTEEREvaluator):
    """Evaluator for LLM fine-tuning implementations with simplified validation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """Evaluate LLM fine-tuning implementation with two-step validation"""

        task_info = target_task.get_task_information()

        # Check task history
        if queried_knowledge is not None:
            if task_info in queried_knowledge.success_task_to_knowledge_dict:
                return queried_knowledge.success_task_to_knowledge_dict[task_info].feedback
            elif task_info in queried_knowledge.failed_task_info_set:
                return CoSTEERSingleFeedback(
                    execution="Task failed too many times, skipping.",
                    return_checking="Task failed too many times, skipping.",
                    code="Task failed too many times, skipping.",
                    final_decision=False,
                )

        env = get_ft_env(
            running_timeout_period=self.scen.real_debug_timeout() if hasattr(self.scen, "real_debug_timeout") else 3600,
        )
        config_yaml = implementation.file_dict.get(FT_YAML_FILE_NAME, "")
        if not config_yaml:
            return CoSTEERSingleFeedback(
                execution=f"No {FT_YAML_FILE_NAME} found",
                return_checking="Configuration file missing",
                code="No valid configuration file",
                final_decision=False,
            )

        # Two-step validation: parameter filtering + micro-batch test
        validation_result = LLMConfigValidator().validate_and_test(
            config_yaml=config_yaml, workspace=implementation, env=env
        )

        # Update config with filtered version
        if validation_result.filtered_config != config_yaml:
            implementation.inject_files(**{FT_YAML_FILE_NAME: validation_result.filtered_config})

        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[target_task.get_task_information()]
            if queried_knowledge is not None
            else []
        )

        system_prompt = T(".prompts:finetune_eval.system").r(
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
        )
        user_prompt = T(".prompts:finetune_eval.user").r(
            scenario=self.scen.get_scenario_all_desc(),
            task_desc=target_task.get_task_information(),
            stdout=validation_result.execution_output or "No output",
            code_yaml=implementation.file_dict[FT_YAML_FILE_NAME],
            workspace_files="\n".join(
                [
                    f"- {file.name} ({file.stat().st_size} bytes)"
                    for file in implementation.workspace_path.rglob("*")
                    if file.is_file() and "checkpoint" not in file.absolute().as_posix()
                ]
            ),
        )
        return build_cls_from_json_with_retry(
            CoSTEERSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=CoSTEERSingleFeedback.val_and_update_init_dict,
        )
