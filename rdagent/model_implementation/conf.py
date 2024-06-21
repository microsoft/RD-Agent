from pathlib import Path
from pydantic_settings import BaseSettings

class ModelImplSettings(BaseSettings):
    workspace_path: Path = Path("./git_ignore_folder/model_imp_workspace/")  # Added type annotation for work_space
    
    class Config:
        env_prefix = 'MODEL_IMPL_'  # Use MODEL_IMPL_ as prefix for environment variables

MODEL_IMPL_SETTINGS = ModelImplSettings()
