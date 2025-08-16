"""
Code generator specifically designed for LLM fine-tuning tasks
"""

import json
import re
from typing import Any, Dict, List

from rdagent.components.coder.data_science.pipeline import PipelineCoSTEER
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.finetune.tasks import DataFormatTask, FineTuningTask
from rdagent.utils.agent.tpl import T

from .utils import create_parameter_validator


class LLMPipelineCoSTEER(PipelineCoSTEER):
    """LLM fine-tuning specific code generator for data processing and fine-tuning tasks"""

    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.parameter_validator = create_parameter_validator()
        logger.info("Initialized LLM Pipeline CoSTEER with parameter validator")

    def develop(self, exp: DSExperiment) -> DSExperiment:
        """Develop LLM fine-tuning experiment"""

        if not exp.pending_tasks_list:
            logger.warning("No pending tasks in experiment")
            return exp

        # Process task list
        for tasks in exp.pending_tasks_list:
            if not tasks:
                continue

            task = tasks[0]  # Simplified: process one task at a time
            logger.info(f"Developing task: {task.name}")

            # Generate code based on task type
            if "DataFormat" in task.name:
                workspace = self._generate_data_format_code(task)
            elif "FineTuning" in task.name:
                workspace = self._generate_finetuning_code(task)
            else:
                logger.warning(f"Unknown task type: {task.name}")
                workspace = self._generate_generic_code(task)

            # Set experiment workspace
            exp.experiment_workspace = workspace
            exp.sub_tasks = tasks

        return exp

    def _generate_data_format_code(self, task: Task) -> FBWorkspace:
        """Generate data format conversion code"""
        logger.info("Generating data format conversion code")

        # Generate prompts from templates
        system_prompt = T("scenarios.finetune.train.prompts:data_format_system_prompt").r()
        user_prompt = T("scenarios.finetune.train.prompts:data_format_user_prompt").r(task_description=task.description)

        # Call LLM to generate code
        response = self._call_llm(system_prompt, user_prompt)
        code = self._extract_code_from_response(response)

        # Create workspace
        workspace = FBWorkspace()
        workspace.inject_files(**{"main.py": code})

        return workspace

    def _generate_finetuning_code(self, task: Task) -> FBWorkspace:
        """Generate fine-tuning code"""
        logger.info("Generating fine-tuning code")

        # Generate prompts from templates
        system_prompt = T("scenarios.finetune.train.prompts:finetuning_system_prompt").r()
        user_prompt = T("scenarios.finetune.train.prompts:finetuning_user_prompt").r(task_description=task.description)

        # Call LLM to generate code
        response = self._call_llm(system_prompt, user_prompt)
        code = self._extract_code_from_response(response)

        # Validate and filter LLaMA Factory parameters in the generated code
        validated_code = self._validate_llamafactory_config_in_code(code)

        # Create workspace
        workspace = FBWorkspace()
        workspace.inject_files(**{"main.py": validated_code})

        return workspace

    def _generate_generic_code(self, task: Task) -> FBWorkspace:
        """Generate generic code"""
        logger.info("Generating generic code")

        # Generate prompts from templates
        system_prompt = T("scenarios.finetune.train.prompts:generic_system_prompt").r()
        user_prompt = T("scenarios.finetune.train.prompts:generic_user_prompt").r(task_description=task.description)

        response = self._call_llm(system_prompt, user_prompt)
        code = self._extract_code_from_response(response)

        workspace = FBWorkspace()
        workspace.inject_files(**{"main.py": code})

        return workspace

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM to generate code"""
        from rdagent.oai.llm_utils import APIBackend

        try:
            api = APIBackend()
            response = api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt, json_mode=False
            )
            return response
        except Exception as e:
            logger.error(f"Failed to call LLM: {e}")
            return self._get_fallback_code()

    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from LLM response"""
        # Find code blocks

        # Find ```python...``` format code blocks
        pattern = r"```python\s*\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            return match.group(1).strip()

        # Find ```...``` format code blocks
        pattern = r"```\s*\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            return match.group(1).strip()

        # If no code block found, return entire response
        logger.warning("No code block found in LLM response")
        return response.strip()

    def _validate_llamafactory_config_in_code(self, code: str) -> str:
        """Validate and filter LLaMA Factory configuration in the generated code."""
        try:
            import yaml

            # Find YAML configurations in the code
            yaml_patterns = [
                # Direct YAML content in multi-line strings
                r'"""([^"]*(?:model_name_or_path|dataset|lora_rank)[^"]*)"""',
                r"'''([^']*(?:model_name_or_path|dataset|lora_rank)[^']*)'''",
                # YAML configurations in variables or file writes
                r'config\s*=\s*[\'"]([^\'"]*(?:model_name_or_path|dataset|lora_rank)[^\'"]*)[\'"]',
                # YAML content being written to files
                r'\.write\([\'"]([^\'"]*(?:model_name_or_path|dataset|lora_rank)[^\'"]*)[\'"]',
                # YAML dump or save operations
                r"yaml\.(?:dump|safe_dump)\s*\(\s*([^)]*)",
            ]

            modified_code = code
            config_validated = False

            for pattern in yaml_patterns:
                matches = re.finditer(pattern, code, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        yaml_content = match.group(1)
                        # Try to parse as YAML
                        config_dict = yaml.safe_load(yaml_content)
                        if isinstance(config_dict, dict):
                            # Validate using parameter validator
                            validated_config = self.parameter_validator.validate_config_dict(config_dict)
                            if validated_config != config_dict:
                                logger.info("LLaMA Factory configuration validated and filtered")
                                validated_yaml = yaml.dump(
                                    validated_config,
                                    default_flow_style=False,
                                    sort_keys=False,
                                )
                                modified_code = modified_code.replace(yaml_content, validated_yaml)
                                config_validated = True
                    except yaml.YAMLError:
                        # Not valid YAML, continue
                        continue
                    except Exception as e:
                        logger.warning(f"Error validating YAML config: {e}")
                        continue

            if config_validated:
                logger.info("Applied LLaMA Factory parameter validation to generated code")

            return modified_code

        except Exception as e:
            logger.error(f"Error validating LLaMA Factory config in code: {e}")
            return code

    def _get_fallback_code(self) -> str:
        """Get fallback code"""
        return """
import os
import sys

def main():
    print("Task execution started...")
    
    # This is a fallback script used when LLM call fails
    print("Warning: This is a fallback script. LLM code generation failed.")
    print("Please check the logs and manual implementation may be needed.")
    
    print("Task execution completed.")

if __name__ == "__main__":
    main()
"""
