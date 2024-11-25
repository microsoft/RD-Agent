from typing import Union

from rdagent.core.conf import ExtendedBaseSettings


class CoSTEERSettings(ExtendedBaseSettings):
    """CoSTEER settings, this setting is supposed not to be used directly!!!"""

    class Config:
        env_prefix = "CoSTEER_"

    coder_use_cache: bool = False
    """Indicates whether to use cache for the coder"""

    max_loop: int = 10
    """Maximum number of task implementation loops"""

    fail_task_trial_limit: int = 20

    v1_query_former_trace_limit: int = 5
    v1_query_similar_success_limit: int = 5

    v2_query_component_limit: int = 1
    v2_query_error_limit: int = 1
    v2_query_former_trace_limit: int = 1
    v2_add_fail_attempt_to_latest_successful_execution: bool = False
    v2_error_summary: bool = False
    v2_knowledge_sampler: float = 1.0

    knowledge_base_path: Union[str, None] = None
    """Path to the knowledge base"""

    new_knowledge_base_path: Union[str, None] = None
    """Path to the new knowledge base"""

    select_threshold: int = 10


CoSTEER_SETTINGS = CoSTEERSettings()
