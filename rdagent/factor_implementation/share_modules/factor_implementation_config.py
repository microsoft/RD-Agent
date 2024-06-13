from pathlib import Path

from pydantic_settings import BaseSettings

from typing import Literal, Union

SELECT_METHOD = Literal["random", "scheduler"]


class FactorImplementSettings(BaseSettings):
    file_based_execution_data_folder: str = str(
        (Path().cwd() / "git_ignore_folder" / "factor_implementation_source_data").absolute(),
    )
    file_based_execution_workspace: str = str(
        (Path().cwd() / "git_ignore_folder" / "factor_implementation_workspace").absolute(),
    )
    implementation_execution_cache_location: str = str(
        (Path().cwd() / "git_ignore_folder" / "factor_implementation_execution_cache.pkl").absolute(),
    )
    enable_execution_cache: bool = True  # whether to enable the execution cache

    # TODO: the factor implement specific settings should not appear in this settings
    # Evolving should have a method specific settings
    # evolving related config
    fail_task_trial_limit: int = 20

    v1_query_former_trace_limit: int = 5
    v1_query_similar_success_limit: int = 5

    v2_query_component_limit: int = 1
    v2_query_error_limit: int = 1
    v2_query_former_trace_limit: int = 1
    v2_error_summary: bool = False
    v2_knowledge_sampler: float = 1.0

    evo_multi_proc_n: int = 16  # how many processes to use for evolving (including eval & generation)

    file_based_execution_timeout: int = 120  # seconds for each factor implementation execution

    select_method: SELECT_METHOD = "random"
    select_ratio: float = 0.5

    max_loop: int = 10

    knowledge_base_path: Union[str, None] = None
    new_knowledge_base_path: Union[str, None] = None

    chat_token_limit: int = (
        100000  # 100000 is the maximum limit of gpt4, which might increase in the future version of gpt
    )


FIS = FactorImplementSettings()
