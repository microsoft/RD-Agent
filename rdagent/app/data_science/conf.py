from typing import Literal

from pydantic_settings import SettingsConfigDict

from rdagent.app.kaggle.conf import KaggleBasePropSetting


class DataScienceBasePropSetting(KaggleBasePropSetting):
    # TODO: Kaggle Setting should be the subclass of DataScience
    model_config = SettingsConfigDict(env_prefix="DS_", protected_namespaces=())

    # Main components
    ## Scen
    scen: str = "rdagent.scenarios.data_science.scen.KaggleScen"
    """
    Scenario class for data science tasks.
    - For Kaggle competitions, use: "rdagent.scenarios.data_science.scen.KaggleScen"
    - For custom data science scenarios, use: "rdagent.scenarios.data_science.scen.DataScienceScen"
    """

    hypothesis_gen: str = "rdagent.scenarios.data_science.proposal.exp_gen.proposal.DSProposalV2ExpGen"
    """Hypothesis generation class"""

    summarizer: str = "rdagent.scenarios.data_science.dev.feedback.DSExperiment2Feedback"
    summarizer_init_kwargs: dict = {
        "version": "exp_feedback",
    }
    ## Workflow Related
    consecutive_errors: int = 5

    ## Coding Related
    coding_fail_reanalyze_threshold: int = 3

    debug_timeout: int = 600
    """The timeout limit for running on debugging data"""
    full_timeout: int = 3600
    """The timeout limit for running on full data"""

    ### specific feature

    #### enable specification
    spec_enabled: bool = True

    #### proposal related
    # proposal_version: str = "v2" deprecated

    coder_on_whole_pipeline: bool = True
    max_trace_hist: int = 3

    coder_max_loop: int = 10
    runner_max_loop: int = 3

    sample_data_by_LLM: bool = True
    use_raw_description: bool = False
    show_nan_columns: bool = False

    #### model dump
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

    #### Evaluation on Test related
    eval_sub_dir: str = "eval"  # TODO: fixme, this is not a good name
    """We'll use f"{DS_RD_SETTING.local_data_path}/{DS_RD_SETTING.eval_sub_dir}/{competition}"
    to find the scriipt to evaluate the submission on test"""

    """---below are the settings for multi-trace---"""

    ### multi-trace related
    max_trace_num: int = 3
    """The maximum number of traces to grow before merging"""

    #### multi-trace:checkpoint selector
    selector_name: str = "rdagent.scenarios.data_science.proposal.exp_gen.select.expand.LatestCKPSelector"
    """The name of the selector to use"""
    sota_count_window: int = 5
    """The number of trials to consider for SOTA count"""
    sota_count_threshold: int = 1
    """The threshold for SOTA count"""

    #### multi-trace: SOTA experiment selector
    sota_exp_selector_name: str = "rdagent.scenarios.data_science.proposal.exp_gen.select.submit.GlobalSOTASelector"
    """The name of the SOTA experiment selector to use"""

    ### multi-trace:inject optimals for multi-trace
    # inject diverse when start a new sub-trace
    enable_inject_diverse: bool = False

    # inject knowledge at the root of the trace
    enable_inject_knowledge_at_root: bool = False

    # enable different version of DSExpGen for multi-trace
    enable_multi_version_exp_gen: bool = False
    exp_gen_version_list: str = "v3,v2"

    #### multi-trace: time for final multi-trace merge
    merge_hours: int = 2
    """The time for merge"""

    #### multi-trace: max SOTA-retrieved number, used in AutoSOTAexpSelector
    # constrains the number of SOTA experiments to retrieve, otherwise too many SOTA experiments to retrieve will cause the exceed of the context window of LLM
    max_sota_retrieved_num: int = 10
    """The maximum number of SOTA experiments to retrieve in a LLM call"""


DS_RD_SETTING = DataScienceBasePropSetting()
