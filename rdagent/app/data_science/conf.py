from pathlib import Path
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

    planner: str = "rdagent.scenarios.data_science.proposal.exp_gen.planner.DSExpPlannerHandCraft"
    hypothesis_gen: str = "rdagent.scenarios.data_science.proposal.exp_gen.router.ParallelMultiTraceExpGen"
    interactor: str = "rdagent.components.interactor.SkipInteractor"
    trace_scheduler: str = "rdagent.scenarios.data_science.proposal.exp_gen.trace_scheduler.RoundRobinScheduler"
    """Hypothesis generation class"""

    summarizer: str = "rdagent.scenarios.data_science.dev.feedback.DSExperiment2Feedback"
    summarizer_init_kwargs: dict = {
        "version": "exp_feedback",
    }
    ## Workflow Related
    consecutive_errors: int = 5

    ## Coding Related
    coding_fail_reanalyze_threshold: int = 3

    debug_recommend_timeout: int = 600
    """The recommend time limit for running on debugging data"""
    debug_timeout: int = 600
    """The timeout limit for running on debugging data"""
    full_recommend_timeout: int = 3600
    """The recommend time limit for running on full data"""
    full_timeout: int = 3600
    """The timeout limit for running on full data"""

    #### model dump
    enable_model_dump: bool = False
    enable_doc_dev: bool = False
    model_dump_check_level: Literal["medium", "high"] = "medium"

    #### MCP documentation search integration
    enable_mcp_documentation_search: bool = False
    """Enable MCP documentation search for error resolution. Requires MCP_ENABLED=true and MCP_CONTEXT7_ENABLED=true in environment."""

    ### specific feature

    ### notebook integration
    enable_notebook_conversion: bool = False

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
    max_trace_num: int = 1
    """The maximum number of traces to grow before merging"""

    scheduler_temperature: float = 1.0
    """The temperature for the trace scheduler for softmax calculation, used in ProbabilisticScheduler"""

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

    # inject diverse from other traces when start a new sub-trace
    enable_cross_trace_diversity: bool = True
    """Enable cross-trace diversity injection when starting a new sub-trace.
    This is different from `enable_inject_diverse` which is for non-parallel cases."""

    diversity_injection_strategy: str = (
        "rdagent.scenarios.data_science.proposal.exp_gen.diversity_strategy.InjectUntilSOTAGainedStrategy"
    )
    """The strategy to use for injecting diversity context."""

    # enable different version of DSExpGen for multi-trace
    enable_multi_version_exp_gen: bool = False
    exp_gen_version_list: str = "v3,v2"

    #### multi-trace: time for final multi-trace merge
    merge_hours: float = 0
    """The time for merge"""

    #### multi-trace: max SOTA-retrieved number, used in AutoSOTAexpSelector
    # constrains the number of SOTA experiments to retrieve, otherwise too many SOTA experiments to retrieve will cause the exceed of the context window of LLM
    max_sota_retrieved_num: int = 10
    """The maximum number of SOTA experiments to retrieve in a LLM call"""

    #### enable draft before first sota experiment
    enable_draft_before_first_sota: bool = False
    enable_planner: bool = False

    model_architecture_suggestion_time_percent: float = 0.75
    allow_longer_timeout: bool = False
    coder_enable_llm_decide_longer_timeout: bool = False
    runner_enable_llm_decide_longer_timeout: bool = False
    coder_longer_timeout_multiplier_upper: int = 3
    runner_longer_timeout_multiplier_upper: int = 2
    coder_timeout_increase_stage: float = 0.3
    runner_timeout_increase_stage: float = 0.3
    runner_timeout_increase_stage_patience: int = 2
    """Number of failures tolerated before escalating to next timeout level (stage width). Every 'patience' failures, timeout increases by 'runner_timeout_increase_stage'"""
    show_hard_limit: bool = True

    #### enable runner code change summary
    runner_enable_code_change_summary: bool = True

    ### Proposal workflow related

    #### Hypothesis Generate related
    enable_simple_hypothesis: bool = False
    """If true, generate simple hypothesis, no more than 2 sentences each."""

    enable_generate_unique_hypothesis: bool = False
    """Enable generate unique hypothesis. If True, generate unique hypothesis for each component. If False, generate unique hypothesis for each component."""

    #### hypothesis critique and rewrite
    enable_hypo_critique_rewrite: bool = False
    """Enable hypothesis critique and rewrite stages for improving hypothesis quality"""
    enable_scale_check: bool = False

    ##### select related
    ratio_merge_or_ensemble: int = 70
    """The ratio of merge or ensemble to be considered as a valid solution"""
    llm_select_hypothesis: bool = False
    """Whether to use LLM to select hypothesis. If True, use LLM selection; if False, use the existing ranking method."""

    #### Task Generate related
    fix_seed_and_data_split: bool = False

    ensemble_time_upper_bound: bool = False

    user_interaction_wait_seconds: int = 6000  # seconds to wait for user interaction
    user_interaction_mid_folder: Path = Path.cwd() / "git_ignore_folder" / "RD-Agent_user_interaction"


DS_RD_SETTING = DataScienceBasePropSetting()

# enable_cross_trace_diversity and llm_select_hypothesis should not be true at the same time
assert not (
    DS_RD_SETTING.enable_cross_trace_diversity and DS_RD_SETTING.llm_select_hypothesis
), "enable_cross_trace_diversity and llm_select_hypothesis cannot be true at the same time"
