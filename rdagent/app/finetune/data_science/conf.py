import os
import torch
from pydantic_settings import SettingsConfigDict

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.conf import RD_AGENT_SETTINGS, ExtendedBaseSettings


class DSFinetuneScen(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="FT_", protected_namespaces=())
    scen: str = "rdagent.app.finetune.data_science.scen.DSFinetuneScen"
    """
    Scenario class for data science tasks.
    - For Kaggle competitions, use: "rdagent.scenarios.data_science.scen.KaggleScen"
    - For custom data science scenarios, use: "rdagent.scenarios.data_science.scen.DataScienceScen"
    - For LLM finetune scenarios, use: "rdagent.app.finetune.llm.scen.LLMFinetuneScen"
    - For Data science finetune scenarios, use: "rdagent.app.finetune.data_science.scen.DSFinetuneScen"
    """

    debug_timeout: int = 3600
    """The timeout limit for running on debugging data"""
    full_timeout: int = 10800
    """The timeout limit for running on full data"""

    coder_on_whole_pipeline: bool = True
    enable_model_dump: bool = True
    app_tpl: str = "app/finetune/data_science/tpl"


def update_settings(competition: str):
    """
    Update the RD_AGENT_SETTINGS with the values from DS_FINETUNE_SETTINGS.
    """
    DS_FINETUNE_SETTINGS = DSFinetuneScen()
    RD_AGENT_SETTINGS.app_tpl = DS_FINETUNE_SETTINGS.app_tpl
    os.environ["DS_CODER_COSTEER_EXTRA_EVALUATOR"] = '["rdagent.app.finetune.share.eval.PrevModelLoadEvaluator"]'
    for field_name, new_value in DS_FINETUNE_SETTINGS.model_dump().items():
        if hasattr(DS_RD_SETTING, field_name):
            setattr(DS_RD_SETTING, field_name, new_value)
    DS_RD_SETTING.competition = competition

def get_training_config():
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32 if torch.cuda.is_available() else 16,
        "use_mixed_precision": True if torch.cuda.is_available() else False,
        "num_workers": 4 if torch.cuda.is_available() else 2,
        "pin_memory": True if torch.cuda.is_available() else False
    }

class GPUConfig:
    @staticmethod
    def setup_cuda_optimizations():
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
    @staticmethod
    def get_optimized_batch_size(base_batch_size=32):
        if torch.cuda.is_available():
            # Adjust based on available GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory > 8e9:  # 8GB
                return base_batch_size * 4
            elif gpu_memory > 4e9:  # 4GB
                return base_batch_size * 2
        return base_batch_size
    
def get_gpu_enhanced_config():
    """Get configuration optimized for GPU if available"""
    gpu_available = torch.cuda.is_available()
    
    return {
        "training": {
            "device": "cuda" if gpu_available else "cpu",
            "use_amp": gpu_available,  
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0
        },
        "data": {
            "num_workers": 4 if gpu_available else 2,
            "pin_memory": gpu_available,
            "prefetch_factor": 2 if gpu_available else 1
        },
        "model": {
            "use_compile": gpu_available,  
            "optimize_for_inference": gpu_available
        }
    }