from pathlib import Path
from typing import Union

from pydantic_settings import BaseSettings


class ModelImplSettings(BaseSettings):
    workspace_path: Path = Path("./git_ignore_folder/model_imp_workspace/")  # Added type annotation for work_space

    class Config:
        env_prefix = "MODEL_IMPL_"  # Use MODEL_IMPL_ as prefix for environment variables

    knowledge_base_path: Union[str, None] = None
    new_knowledge_base_path: Union[str, None] = None

    max_loop: int = 10

    query_former_trace_limit: int = 5
    query_similar_success_limit: int = 5
    fail_task_trial_limit: int = 20

    evo_multi_proc_n: int = 1


MODEL_IMPL_SETTINGS = ModelImplSettings()
