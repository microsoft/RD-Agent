"""
LLM Fine-tuning Evaluation Components

Provides evaluation functionality for LLM fine-tuning tasks,
including AIME and other math benchmark evaluations.
"""

import json
from pathlib import Path
from typing import Any, Dict

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

DIRNAME = Path(__file__).absolute().resolve().parent


class LLMFinetuneEvaluator(CoSTEEREvaluator):
    """Evaluator for LLM fine-tuning implementations"""

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """Evaluate LLM fine-tuning implementation"""

        task_info = target_task.get_task_information()

        # Check if task already succeeded or failed too many times
        if queried_knowledge is not None:
            if task_info in queried_knowledge.success_task_to_knowledge_dict:
                return queried_knowledge.success_task_to_knowledge_dict[task_info].feedback
            elif task_info in queried_knowledge.failed_task_info_set:
                return CoSTEERSingleFeedback(
                    execution="Task has failed too many times, skipping implementation.",
                    return_checking="Task has failed too many times, skipping implementation.",
                    code="Task has failed too many times, skipping implementation.",
                    final_decision=False,
                )

        # Get fine-tuning environment
        env = get_ft_env(
            running_timeout_period=self.scen.real_debug_timeout() if hasattr(self.scen, "real_debug_timeout") else 3600,
        )

        # Test LlamaFactory config
        test_code = self._generate_config_test_code()
        implementation.inject_files(**{"test_config.py": test_code})

        result = implementation.run(env=env, entry="python test_config.py")
        stdout = result.stdout
        ret_code = result.exit_code

        stdout += f"\nReturn code: {ret_code}"

        # Generate evaluation feedback using LLM
        system_prompt = T(".prompts:finetune_eval.system").r(
            task_desc=task_info,
            test_code=test_code,
            code=implementation.file_dict.get("config.yaml", "No config.yaml found"),
        )

        user_prompt = T(".prompts:finetune_eval.user").r(
            stdout=stdout,
        )

        try:
            feedback = build_cls_from_json_with_retry(
                CoSTEERSingleFeedback,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                init_kwargs_update_func=CoSTEERSingleFeedback.val_and_update_init_dict,
            )
            feedback.final_decision = feedback.final_decision and ret_code == 0
            return feedback
        except Exception as e:
            logger.error(f"Failed to generate evaluation feedback: {e}")
            return CoSTEERSingleFeedback(
                execution=f"Evaluation failed: {str(e)}",
                return_checking="Could not evaluate return values due to evaluation failure.",
                code="Could not evaluate code quality due to evaluation failure.",
                final_decision=False,
            )

    def _generate_config_test_code(self) -> str:
        """Generate test code for LlamaFactory config validation"""
        return '''
"""
Test script for LlamaFactory configuration validation
"""

import sys
import os
import yaml
from pathlib import Path

def test_config_yaml():
    """Test if the config.yaml is valid and contains required parameters"""
    
    print("=== LlamaFactory Config Test ===")
    
    try:
        # Test file existence
        config_path = Path("config.yaml")
        if not config_path.exists():
            print("✗ config.yaml not found")
            return False
        
        print("✓ config.yaml exists")
        
        # Test YAML parsing
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        print("✓ config.yaml is valid YAML")
        
        # Check required fields
        required_fields = [
            "model_name_or_path",
            "stage", 
            "do_train",
            "finetuning_type",
            "dataset",
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in config:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"✗ Missing required fields: {missing_fields}")
            return False
        
        print(f"✓ All required fields present: {required_fields}")
        
        # Check finetuning method specific parameters
        finetuning_type = config.get("finetuning_type", "")
        if finetuning_type in ["lora", "qlora"]:
            lora_fields = ["lora_rank", "lora_alpha"]
            missing_lora = [f for f in lora_fields if f not in config]
            if missing_lora:
                print(f"⚠ Missing LoRA fields for {finetuning_type}: {missing_lora}")
            else:
                print(f"✓ LoRA configuration complete for {finetuning_type}")
        
        # Validate parameter values
        if config.get("max_samples") and config["max_samples"] == 100:
            print("✓ Debug mode detected (max_samples=100)")
        elif config.get("max_samples") is None:
            print("✓ Full training mode detected (no max_samples limit)")
        
        # Test LlamaFactory parameter validation
        try:
            from llamafactory.hparams.data_args import DataArguments
            from llamafactory.hparams.model_args import ModelArguments
            from llamafactory.hparams.finetuning_args import FinetuningArguments
            print("✓ LlamaFactory modules can be imported")
        except ImportError as e:
            print(f"⚠ LlamaFactory import failed (may be expected in test env): {e}")
        
        print("=== Config Validation Completed ===")
        print(f"Configuration summary:")
        print(f"  Model: {config.get('model_name_or_path', 'Unknown')}")
        print(f"  Method: {config.get('finetuning_type', 'Unknown')}")
        print(f"  Dataset: {config.get('dataset', 'Unknown')}")
        print(f"  Max Samples: {config.get('max_samples', 'Unlimited')}")
        
        return True
        
    except yaml.YAMLError as e:
        print(f"✗ YAML parsing error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_config_yaml()
    sys.exit(0 if success else 1)
'''
