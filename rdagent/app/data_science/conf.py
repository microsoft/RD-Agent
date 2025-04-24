from typing import Literal

from pydantic_settings import SettingsConfigDict

from rdagent.app.kaggle.conf import KaggleBasePropSetting


class DataScienceBasePropSetting(KaggleBasePropSetting):
    model_config = SettingsConfigDict(env_prefix="DS_", protected_namespaces=())

    # Main components
    ## Scen
    scen: str = "rdagent.scenarios.data_science.scen.KaggleScen"
    """Scenario class for data mining model"""

    ## Workflow Related
    consecutive_errors: int = 5

    debug_timeout: int = 600
    """The timeout limit for running on debugging data"""
    full_timeout: int = 3600
    """The timeout limit for running on full data"""

    ### specific feature

    #### enable specification
    spec_enabled: bool = True

    ### proposal related
    proposal_version: str = "v1"
    coder_on_whole_pipeline: bool = False
    max_trace_hist: int = 3

    coder_max_loop: int = 10
    runner_max_loop: int = 1

    rule_base_eval: bool = False

    ### model dump
    enable_model_dump: bool = False
    enable_doc_dev: bool = False
    model_dump_check_level: Literal["medium", "high"] = "medium"

    ### knowledge base
    enable_knowledge_base: bool = False
    knowledge_base_version: str = "v1"
    knowledge_base_path: str | None = None
    idea_pool_json_path: str | None = None

    ### archive log folder after each loop
    enable_log_archive: bool = True
    log_archive_path: str | None = None
    log_archive_temp_path: str | None = (
        None  # This is to store the mid tar file since writing the tar file is preferred in local storage then copy to target storage
    )


DS_RD_SETTING = DataScienceBasePropSetting()
