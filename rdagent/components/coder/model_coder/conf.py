from pathlib import Path
from typing import Union

from pydantic_settings import BaseSettings


class ModelImplSettings(BaseSettings):
    class Config:
        env_prefix = "MODEL_CODER_"  # Use MODEL_CODER_ as prefix for environment variables

    coder_use_cache: bool = False

    knowledge_base_path: Union[str, None] = None
    new_knowledge_base_path: Union[str, None] = None

    max_loop: int = 10

    query_former_trace_limit: int = 5
    query_similar_success_limit: int = 5
    fail_task_trial_limit: int = 20


MODEL_IMPL_SETTINGS = ModelImplSettings()
