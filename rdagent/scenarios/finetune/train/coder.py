"""
Code generator specifically designed for LLM fine-tuning tasks
"""

import json
from typing import Any, Dict, List

from rdagent.components.coder.data_science.pipeline import PipelineCoSTEER
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.finetune.tasks import DataFormatTask, FineTuningTask
from rdagent.utils.agent.tpl import T


class LLMPipelineCoSTEER(PipelineCoSTEER):
    """LLM fine-tuning specific code generator for data processing and fine-tuning tasks"""

    def __init__(self, scen: Scenario):
        super().__init__(scen)
        logger.info("Initialized LLM Pipeline CoSTEER")

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

        # Use LLM to generate data processing code
        system_prompt = """
        You are a data processing expert. You need to write a Python script based on user requirements to process datasets
        and convert them to LLaMA-Factory compatible format.
        
        Please ensure:
        1. Code can run directly
        2. All necessary imports are included
        3. Proper error handling
        4. Clear logging output
        5. Save files to specified paths
        """

        user_prompt = f"""
        Please generate a complete Python script (main.py) based on the following task description:
        
        {task.description}
        
        Output format:
        ```python
        # main.py - Data processing script
        [Complete Python code]
        ```
        """

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

        system_prompt = """
        You are an LLM fine-tuning expert. You need to use the LLaMA-Factory framework to fine-tune large language models.
        
        Please ensure:
        1. Use llamafactory-cli commands for training
        2. Properly configure training parameters
        3. Set appropriate LoRA parameters
        4. Include training monitoring and logging
        5. Save the fine-tuned model
        
        CRITICAL: Only use officially supported LLaMA-Factory parameters. Do NOT include these unsupported parameters:
        - merge_lora_after_train
        - merge_lora
        - auto_merge_lora
        
        If LoRA merging is needed, handle it as a separate post-training step, not in the training configuration.
        """

        user_prompt = f"""
        Please generate a complete Python script (main.py) based on the following task description:
        
        {task.description}
        
        Output format:
        ```python
        # main.py - LLM fine-tuning script
        [Complete Python code]
        ```
        """

        # Call LLM to generate code
        response = self._call_llm(system_prompt, user_prompt)
        code = self._extract_code_from_response(response)

        # Create workspace
        workspace = FBWorkspace()
        workspace.inject_files(**{"main.py": code})

        return workspace

    def _generate_generic_code(self, task: Task) -> FBWorkspace:
        """Generate generic code"""
        logger.info("Generating generic code")

        system_prompt = "You are a programming expert, generate Python code based on task description."
        user_prompt = f"Task description: {task.description}\n\nPlease generate a complete Python script."

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
        import re

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
