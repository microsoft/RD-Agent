from pathlib import Path
from typing import Literal, Union

from pydantic_settings import BaseSettings

SELECT_METHOD = Literal["random", "scheduler"]


class FactorImplementSettings(BaseSettings):
    class Config:
        env_prefix = "FACTOR_CODER_"
        """Use `FACTOR_CODER_` as prefix for environment variables"""

    coder_use_cache: bool = False
    """Indicates whether to use cache for the coder"""

    data_folder: str = "git_ignore_folder/factor_implementation_source_data"
    """Path to the folder containing financial data (default is fundamental data in Qlib)"""

    data_folder_debug: str = "git_ignore_folder/factor_implementation_source_data_debug"
    """Path to the folder containing partial financial data (for debugging)"""

    cache_location: str = "git_ignore_folder/factor_implementation_execution_cache"
    """Path to the cache location"""

    enable_execution_cache: bool = True
    """Indicates whether to enable the execution cache"""

    # TODO: the factor implement specific settings should not appear in this settings
    # Evolving should have a method specific settings
    # evolving related config
    fail_task_trial_limit: int = 20

    v1_query_former_trace_limit: int = 5
    v1_query_similar_success_limit: int = 5

    v2_query_component_limit: int = 1
    v2_query_error_limit: int = 1
    v2_query_former_trace_limit: int = 1
    v2_add_fail_attempt_to_latest_successful_execution: bool = False
    v2_error_summary: bool = False
    v2_knowledge_sampler: float = 1.0

    file_based_execution_timeout: int = 120
    """Timeout in seconds for each factor implementation execution"""

    select_method: str = "random"
    """Method for the selection of factors implementation"""

    select_threshold: int = 10
    """Threshold for the number of factor selections"""

    max_loop: int = 10
    """Maximum number of task implementation loops"""

    knowledge_base_path: Union[str, None] = None
    """Path to the knowledge base"""

    new_knowledge_base_path: Union[str, None] = None
    """Path to the new knowledge base"""

    python_bin: str = "python"
    """Path to the Python binary"""


FACTOR_IMPLEMENT_SETTINGS = FactorImplementSettings()
