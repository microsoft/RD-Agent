from pathlib import Path
from typing import Union

from pydantic_settings import BaseSettings


class ModelImplSettings(BaseSettings):
    class Config:
        env_prefix = "MODEL_IMPL_"  # Use MODEL_IMPL_ as prefix for environment variables

    model_execution_workspace: str = str(
        (Path().cwd() / "git_ignore_folder" / "model_implementation_workspace").absolute(),
    )
    model_cache_location: str = str(
        (Path().cwd() / "git_ignore_folder" / "model_implementation_execution_cache").absolute(),
    )

    knowledge_base_path: Union[str, None] = None
    new_knowledge_base_path: Union[str, None] = None

    max_loop: int = 10

    query_former_trace_limit: int = 5
    query_similar_success_limit: int = 5
    fail_task_trial_limit: int = 20

    evo_multi_proc_n: int = 1

    enable_execution_cache: bool = True  # whether to enable the execution cache


MODEL_IMPL_SETTINGS = ModelImplSettings()
