"""
LLM Fine-tuning Evaluation Components

Provides simplified evaluation: parameter filtering + micro-batch testing.
No redundant LLM feedback generation - test results speak for themselves.
"""

import json
import random
import re
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
    get_workspace_prefix,
)
from rdagent.components.coder.finetune.unified_validator import LLMConfigValidator
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.scen.utils import _compute_column_stats
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

DIRNAME = Path(__file__).absolute().resolve().parent


class FTDataEvaluator(CoSTEEREvaluator):
    """Evaluator for data processing results.

    This evaluator:
    1. Executes the process_data.py script in Docker
    2. Validates the output data.json file
    3. Generates dataset_info.json for LlamaFactory
    """

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: Optional[QueriedKnowledge] = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """Evaluate data processing implementation with LLM feedback."""

        script_code = implementation.file_dict.get(FT_DATA_SCRIPT_NAME, "")
        data_json_path = implementation.workspace_path / FT_DATA_FILE_NAME
        execution_output = ""
        exit_code = 0
        data = None
        error_msg = None

        # Step 1: Check script exists
        if not script_code:
            feedback = CoSTEERSingleFeedback(
                execution=f"No {FT_DATA_SCRIPT_NAME} found",
                return_checking="Data processing script missing",
                code="Please generate a data processing script first.",
                final_decision=False,
            )
            logger.log_object(feedback, tag="evaluator_feedback.FTDataEvaluator")
            return feedback

        # Step 2: Check if data.json already exists
        if data_json_path.exists():
            validation_result = self._validate_data_json(data_json_path)
            if validation_result["valid"]:
                logger.info("Valid data.json already exists, skipping execution")
                self._update_dataset_info(implementation, validation_result["sample_count"])
        else:
            # Step 3: Execute script in DEBUG mode (generates ~10 samples for fast validation)
            env, env_vars = get_data_processing_env()
            ws_prefix = get_workspace_prefix(env)
            try:
                # Use FTWorkspace.run() for unified Docker logging
                # --debug flag tells the script to generate only ~10 samples
                result = implementation.run(
                    env=env,
                    entry=f"python {ws_prefix}/{FT_DATA_SCRIPT_NAME} --debug",
                    env_vars=env_vars,
                )
                execution_output = result.stdout if hasattr(result, "stdout") else str(result)
                exit_code = result.exit_code if hasattr(result, "exit_code") else -1
            except Exception as e:
                logger.error(f"Failed to execute data processing script: {e}")
                feedback = CoSTEERSingleFeedback(
                    execution=f"Script execution failed: {e}",
                    return_checking="Execution error",
                    code="Check script for syntax errors or missing dependencies.",
                    final_decision=False,
                )
                logger.log_object(feedback, tag="evaluator_feedback.FTDataEvaluator")
                return feedback

            # Step 4: Validate output
            if not data_json_path.exists():
                error_msg = f"{FT_DATA_FILE_NAME} not generated"
            else:
                validation_result = self._validate_data_json(data_json_path)
                if not validation_result["valid"]:
                    error_msg = validation_result["error"]
                else:
                    self._update_dataset_info(implementation, validation_result["sample_count"])

        # Step 5: Load data if valid
        if error_msg is None and data_json_path.exists():
            with open(data_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Step 5.5: Compute token stats and inject data_stats for yaml coder
        if data is not None and error_msg is None:
            self._inject_data_stats(implementation, data, execution_output)

        # Step 6: Generate LLM feedback
        # Truncate stdout from end for LLM (summary at the end is more useful)
        stdout_summary = execution_output[-1500:] if execution_output else ""
        return self._generate_llm_feedback(
            target_task=target_task,
            script_code=script_code if error_msg else "",  # Only show script on error
            stdout=stdout_summary,  # Always show summary (truncated from end)
            exit_code=exit_code,
            data=data,
            error_msg=error_msg,
            queried_knowledge=queried_knowledge,
            raw_stdout=execution_output,  # Full log for UI
        )

    def _generate_llm_feedback(
        self,
        target_task: Task,
        script_code: str,
        stdout: str,
        exit_code: int,
        data: Optional[list],
        error_msg: Optional[str],
        queried_knowledge: Optional[QueriedKnowledge],
        raw_stdout: str = "",
    ) -> CoSTEERSingleFeedback:
        """Generate LLM-based feedback for data processing evaluation."""

        # Prepare data statistics and samples
        if data:
            stats = self._analyze_data_quality(data)
            data_stats = json.dumps(stats, indent=2)
            sampled_data = self._sample_data(data)
            data_samples = json.dumps(sampled_data, indent=2, ensure_ascii=False)
            sample_count = len(sampled_data)
            total_samples = len(data)
        else:
            data_stats = json.dumps({"error": error_msg or "No data generated"})
            data_samples = "[]"
            sample_count = 0
            total_samples = 0

        # Extract similar successful knowledge
        queried_similar_successful_knowledge = []
        if queried_knowledge is not None:
            task_info = target_task.get_task_information()
            queried_similar_successful_knowledge = queried_knowledge.task_to_similar_task_successful_knowledge.get(
                task_info, []
            )

        # Build prompts
        system_prompt = T(".prompts:data_eval.system").r(
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
        )
        user_prompt = T(".prompts:data_eval.user").r(
            task_desc=target_task.get_task_information(),
            script_code=script_code,
            exit_code=exit_code,
            stdout=stdout[:3000] if stdout else "",  # Empty string triggers {% if stdout %} = false
            data_stats=data_stats,
            sample_count=sample_count,
            total_samples=total_samples,
            data_samples=data_samples,
        )

        logger.info(
            f"Generating LLM feedback for data evaluation (samples: {total_samples}, has_error: {bool(error_msg)})"
        )

        feedback = build_cls_from_json_with_retry(
            CoSTEERSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=CoSTEERSingleFeedback.val_and_update_init_dict,
        )
        feedback.raw_execution = raw_stdout
        logger.log_object(feedback, tag="evaluator_feedback.FTDataEvaluator")
        return feedback

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

    def _sample_data(self, data: list, n: int = 5) -> list:
        """Random sampling for LLM evaluation."""
        if len(data) <= n:
            return data
        return random.sample(data, n)

    def _analyze_data_quality(self, data: list) -> dict:
        """Analyze data quality statistics for all fields."""
        if not data:
            return {"total_samples": 0, "error": "Empty data"}

        # Analyze length stats for all standard fields
        fields = ["instruction", "input", "output"]
        stats = {"total_samples": len(data)}

        for field in fields:
            lens = [len(str(d.get(field, ""))) for d in data]
            empty_count = sum(1 for d in data if not d.get(field))
            stats[f"{field}_len"] = {
                "min": min(lens),
                "max": max(lens),
                "avg": round(sum(lens) / len(lens), 1),
            }
            stats[f"{field}_empty_ratio"] = round(empty_count / len(data) * 100, 1)

        # Detect duplicates by full record (instruction + input + output)
        record_set = set(
            (str(d.get("instruction", "")), str(d.get("input", "")), str(d.get("output", ""))) for d in data
        )
        duplicate_count = len(data) - len(record_set)
        stats["duplicate_count"] = duplicate_count
        stats["duplicate_ratio"] = round(duplicate_count / len(data) * 100, 1)

        return stats

    def _parse_estimation_from_stdout(self, stdout: str) -> dict:
        """Parse estimation info from script SUMMARY output.

        Expected format in stdout:
        ========== SUMMARY ==========
        Total output samples: 10
        Raw samples processed: 100
        Raw samples total: 50000
        Estimated full output: ~5000
        =============================
        """
        estimation = {}
        if not stdout:
            return estimation

        patterns = {
            "output_samples": r"Total output samples:\s*(\d+)",
            "raw_processed": r"Raw samples processed:\s*(\d+)",
            "raw_total": r"Raw samples total:\s*(\d+)",
            "estimated_full": r"Estimated full output:\s*~?(\d+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, stdout)
            if match:
                estimation[key] = int(match.group(1))

        return estimation

    def _inject_data_stats(self, implementation: FBWorkspace, data: list, stdout: str) -> None:
        """Compute token statistics and inject data_stats.json for yaml coder.

        This method:
        1. Computes token statistics using utils._compute_column_stats
        2. Parses estimation info from script stdout
        3. Injects data_stats.json into workspace.file_dict for yaml coder to read
        """
        try:
            # Compute token statistics (min/max/p50/p99 for each field)
            token_stats = _compute_column_stats(data)

            # Parse estimation from stdout
            estimation = self._parse_estimation_from_stdout(stdout)

            # Combine into data_stats
            data_stats = {
                "total_samples": len(data),
                "token_stats": token_stats,
                "estimation": estimation,
            }

            # Inject into workspace for yaml coder to read
            implementation.inject_files(**{"data_stats.json": json.dumps(data_stats, indent=2)})
            logger.info(f"Injected data_stats.json with token stats for {len(data)} samples")

        except Exception as e:
            logger.warning(f"Failed to inject data_stats: {e}")


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
                feedback = CoSTEERSingleFeedback(
                    execution="Task failed too many times, skipping.",
                    return_checking="Task failed too many times, skipping.",
                    code="Task failed too many times, skipping.",
                    final_decision=False,
                )
                logger.log_object(feedback, tag="evaluator_feedback.FTCoderEvaluator")
                return feedback

        env = get_ft_env(operation="micro_batch")
        config_yaml = implementation.file_dict.get(FT_YAML_FILE_NAME, "")
        if not config_yaml:
            feedback = CoSTEERSingleFeedback(
                execution=f"No {FT_YAML_FILE_NAME} found",
                return_checking="Configuration file missing",
                code="No valid configuration file",
                final_decision=False,
            )
            logger.log_object(feedback, tag="evaluator_feedback.FTCoderEvaluator")
            return feedback

        # Two-step validation: parameter filtering + micro-batch test
        validation_result = LLMConfigValidator().validate_and_test(
            config_yaml=config_yaml, workspace=implementation, env=env
        )
        # NOTE: Docker execution is logged by FTWorkspace.run() automatically

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
        feedback = build_cls_from_json_with_retry(
            CoSTEERSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=CoSTEERSingleFeedback.val_and_update_init_dict,
        )
        feedback.raw_execution = validation_result.raw_stdout or ""
        logger.log_object(feedback, tag="evaluator_feedback.FTCoderEvaluator")
        return feedback
