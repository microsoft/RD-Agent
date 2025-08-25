"""
LLM Fine-tuning CoSTEER Implementation

This module provides fine-tuning specific components for the CoSTEER framework,
including evaluators and evolving strategies.
"""

from pathlib import Path

import yaml

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiEvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.components.coder.data_science.share.ds_costeer import DSCoSTEER
from rdagent.components.coder.finetune.conf import FTCoderCoSTEERSettings
from rdagent.components.coder.finetune.eval import LLMFinetuneEvaluator
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.finetune.train.utils import create_parameter_validator
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent


class LLMFinetuneEvolvingStrategy(MultiProcessEvolvingStrategy):
    """LLM Fine-tuning specific evolving strategy"""

    def __init__(self, scen: Scenario, settings, *args, **kwargs):
        super().__init__(scen, settings)

        # Initialize LlamaFactory parameter validator
        from rdagent.scenarios.finetune.train.utils import create_parameter_validator

        self.parameter_validator = create_parameter_validator()

    def implement_one_task(
        self,
        target_task: Task,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        """Implement a single fine-tuning task by generating LlamaFactory config"""

        task_info = target_task.get_task_information()

        # Query relevant knowledge
        similar_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge.get(task_info, []) if queried_knowledge else []
        )

        failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces.get(task_info, ([], None))
            if queried_knowledge
            else ([], None)
        )

        # Get task parameters from the task object
        base_model = getattr(target_task, "base_model", "Qwen2.5-1.5B-Instruct")
        finetune_method = getattr(target_task, "finetune_method", "lora")
        dataset = getattr(target_task, "dataset", "default")

        # Use LLM to generate LlamaFactory config YAML
        # For coding stage, use debug mode with limited samples
        config_yaml = self._generate_llamafactory_config_with_llm(
            base_model=base_model,
            finetune_method=finetune_method,
            dataset=dataset,
            debug_mode=True,  # Use debug mode for coding stage (limited samples for quick validation)
            task_info=task_info,
            similar_knowledge=similar_knowledge,
            failed_knowledge=failed_knowledge[0],
            prev_feedback=prev_task_feedback,
            workspace=workspace,
        )

        # Validate the generated config using existing validator
        validated_config = self._validate_config(config_yaml)

        return {"config.yaml": validated_config}

    def _generate_llamafactory_config_with_llm(
        self,
        base_model: str,
        finetune_method: str,
        dataset: str,
        debug_mode: bool = True,
        task_info: str = "",
        similar_knowledge: list = None,
        failed_knowledge: list = None,
        prev_feedback=None,
        workspace=None,
    ) -> str:
        """Generate LlamaFactory configuration YAML using LLM"""

        # Prepare knowledge context
        similar_knowledge_str = ""
        if similar_knowledge:
            similar_knowledge_str = "\n".join(
                [
                    f"### Similar Implementation {i+1}:\n{knowledge.target_task.get_task_information()}\n```yaml\n{knowledge.implementation.file_dict.get('config.yaml', '')}\n```"
                    for i, knowledge in enumerate(similar_knowledge)
                ]
            )

        failed_knowledge_str = ""
        if failed_knowledge:
            failed_knowledge_str = "\n".join(
                [
                    f"### Failed Attempt {i+1}:\n```yaml\n{knowledge.implementation.file_dict.get('config.yaml', '')}\n```\n**Feedback:** {knowledge.feedback}"
                    for i, knowledge in enumerate(failed_knowledge)
                ]
            )

        # Query LLaMA Factory parameters for the specific method
        from rdagent.scenarios.finetune.train.llama_params_query import (
            LLaMAFactoryParamsQuery,
        )

        params_query = LLaMAFactoryParamsQuery()
        method_params_desc = params_query.format_params_for_prompt(finetune_method)

        # Generate prompts using templates
        system_prompt = T("components.coder.finetune.prompts:finetune_coder.system").r(
            task_desc=task_info,
            finetune_method=finetune_method,
            similar_knowledge=similar_knowledge if similar_knowledge else [],
            failed_knowledge=failed_knowledge if failed_knowledge else [],
            method_params=method_params_desc,
        )

        # Get Docker workspace information (try actual structure first, fallback to expected)
        shared_workspace_dir = None

        # Try to get shared workspace directory from various sources
        try:
            # First try: check if available from FT_RD_SETTING or environment
            from pathlib import Path

            from rdagent.app.finetune.llm.conf import FT_RD_SETTING

            # Construct expected shared workspace path based on configuration
            if hasattr(FT_RD_SETTING, "local_data_path") and FT_RD_SETTING.local_data_path:
                # Look for shared workspace in the same parent directory as local_data_path
                data_parent = Path(FT_RD_SETTING.local_data_path).parent
                potential_shared = data_parent / "llm_finetune_shared_workspace"
                if potential_shared.exists():
                    shared_workspace_dir = potential_shared
                    logger.info(f"Found shared workspace directory: {shared_workspace_dir}")
        except Exception as e:
            logger.debug(f"Could not auto-detect shared workspace directory: {e}")

        # Generate simplified Docker environment info
        docker_env_info = f"""## Docker Container Environment
**Container Image**: local_llm_finetune:latest
**Mount Path**: /workspace/

**Expected File Structure in Docker Container**:
```
/workspace/                              # Main working directory
├── data/                                # Dataset configuration directory
│   ├── dataset_info.json               # LlamaFactory dataset configuration
│   └── processed_dataset.json          # Preprocessed training data
├── dataset/{dataset}/                   # Raw dataset files
├── shared/                              # Shared data processing outputs
│   ├── dataset_info.json               # Dataset configuration (copy)
│   └── processed_dataset.json          # Preprocessed data (copy)
├── output/                              # Training output directory
└── model/                               # Model files
```

**Critical Docker Paths** (use these exact paths in your configuration):
- Working Directory: `/workspace/`
- Dataset Directory: `/workspace/data`
- Raw Dataset: `/workspace/dataset/{dataset}`
- Processed Dataset: `/workspace/data/processed_dataset.json`
- Dataset Configuration: `/workspace/data/dataset_info.json`
- Output Directory: `/workspace/output`"""

        # Define critical configuration rules
        critical_rules = [
            'dataset: MUST be "processed_dataset" (string, not dictionary)',
            'dataset_dir: MUST be "/workspace/data"',
            'output_dir: MUST be "/workspace/output"',
            "model_name_or_path: Use HuggingFace model identifier",
            "All file paths must use Docker container paths, NOT local filesystem paths",
        ]

        user_prompt = T("components.coder.finetune.prompts:finetune_coder.user").r(
            docker_env_info=docker_env_info,
            critical_rules=critical_rules,
            latest_code=(workspace.file_dict.get("config.yaml", "") if workspace and prev_feedback else ""),
            latest_feedback=str(prev_feedback) if prev_feedback else "",
            finetune_method=finetune_method,
            base_model=base_model,
        )

        # Call LLM to generate config
        try:
            from rdagent.oai.llm_utils import APIBackend

            api = APIBackend()
            response = api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=False,
            )

            # Extract YAML content from response
            import re

            # Try multiple YAML extraction patterns
            yaml_patterns = [
                r"```yaml\s*\n(.*?)\n```",  # Standard markdown yaml block
                r"```\s*\n(.*?)\n```",  # Generic code block
                r"yaml\s*:\s*\n(.*?)(?=\n\S|\Z)",  # yaml: prefix
            ]

            extracted_yaml = None
            for pattern in yaml_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted_yaml = match.group(1).strip()
                    logger.info(f"Extracted YAML using pattern: {pattern}")
                    break

            if extracted_yaml:
                # Validate extracted YAML
                try:
                    import yaml

                    yaml.safe_load(extracted_yaml)
                    return extracted_yaml
                except yaml.YAMLError as e:
                    logger.warning(f"Extracted YAML is invalid: {e}, trying fallback")

            # Fallback: try to find any YAML-like content
            yaml_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*\s*:.*?)(?=\n[a-zA-Z_][a-zA-Z0-9_]*\s*:|$)"
            matches = re.findall(yaml_pattern, response, re.DOTALL | re.MULTILINE)
            if matches:
                fallback_yaml = "\n".join(matches)
                try:
                    import yaml

                    yaml.safe_load(fallback_yaml)
                    return fallback_yaml
                except yaml.YAMLError:
                    logger.warning("Fallback YAML extraction also failed")

            # If no YAML found, return fallback config
            logger.warning("No YAML configuration found in LLM response, using fallback")
            return self._generate_llamafactory_config(
                base_model, finetune_method, dataset, debug_mode, task_info, similar_knowledge, failed_knowledge
            )

        except Exception as e:
            logger.error(f"Failed to generate config with LLM: {e}, using fallback")
            return self._generate_llamafactory_config(
                base_model, finetune_method, dataset, debug_mode, task_info, similar_knowledge, failed_knowledge
            )

    def assign_code_list_to_evo(self, code_list: list[dict[str, str]], evo):
        """Assign generated code to the evolving experiment"""
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = evo.experiment_workspace
            evo.sub_workspace_list[index].inject_files(**code_list[index])
        return evo

    def _generate_llamafactory_config(
        self,
        base_model: str,
        finetune_method: str,
        dataset: str,
        debug_mode: bool = True,
        task_info: str = "",
        similar_knowledge: list = None,
        failed_knowledge: list = None,
    ) -> str:
        """Generate LlamaFactory configuration YAML"""

        # Base configuration
        # Always use processed_dataset for LLM finetune scenario
        # This aligns with the preprocessed data from data processing step
        dataset = "processed_dataset"

        config = {
            "model_name_or_path": base_model,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": finetune_method,
            "dataset": dataset,
            "dataset_dir": "/workspace/data",  # Directory containing dataset_info.json
            "template": "qwen",
            "cutoff_len": 2048,
            "max_samples": 100 if debug_mode else None,  # Debug mode uses only 100 samples
            "overwrite_cache": True,
            "preprocessing_num_workers": 16,
            "output_dir": "/workspace/output",
            "logging_steps": 10,
            "save_steps": 500,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "num_train_epochs": 3.0,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "bf16": True,
            "ddp_timeout": 180000000,
            "val_size": 0.1,
            "eval_strategy": "steps",
            "eval_steps": 250,
            "per_device_eval_batch_size": 1,
            "seed": 42,
        }

        # Add method-specific parameters
        if finetune_method in ["lora", "qlora"]:
            config.update(
                {
                    "lora_rank": 8,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1,
                    "lora_target": "all",
                }
            )

        if finetune_method == "qlora":
            config.update(
                {
                    "quantization_bit": 4,
                    "double_quantization": True,
                    "quantization_type": "nf4",
                }
            )

        # Convert to YAML string
        yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)

        return yaml_content

    def _validate_config(self, config_yaml: str) -> str:
        """Validate LlamaFactory configuration using existing validator"""
        try:
            return self.parameter_validator.validate_yaml_config(config_yaml)
        except Exception as e:
            from rdagent.log import rdagent_logger as logger

            logger.warning(f"Config validation failed: {e}, using original config")
            return config_yaml


class LLMFinetuneCoSTEER(DSCoSTEER):
    """LLM Fine-tuning CoSTEER implementation"""

    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        settings = FTCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator(LLMFinetuneEvaluator(scen=scen), scen=scen)
        es = LLMFinetuneEvolvingStrategy(scen=scen, settings=settings)

        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            max_loop=FT_RD_SETTING.coder_max_loop if hasattr(FT_RD_SETTING, "coder_max_loop") else 5,
            **kwargs,
        )
