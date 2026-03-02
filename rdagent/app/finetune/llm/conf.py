from pathlib import Path

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class LLMFinetunePropSetting(ExtendedBaseSettings):
    """LLM Fine-tune dedicated property settings.

    - Adjust timeouts and template
    - Use FT_ env prefix for overrides
    """

    model_config = SettingsConfigDict(env_prefix="FT_", protected_namespaces=())

    # Main Components
    scen: str = "rdagent.scenarios.finetune.scen.scenario.LLMFinetuneScen"
    """Scenario class for LLM fine-tuning tasks."""

    hypothesis_gen: str = "rdagent.scenarios.finetune.proposal.proposal.LLMFinetuneExpGen"
    """Hypothesis generation class for LLM fine-tuning tasks."""

    coder: str = "rdagent.components.coder.finetune.LLMFinetuneCoSTEER"
    """Code generator.
    Function: Generate LLM fine-tuning code based on experiment design.
    """

    runner: str = "rdagent.scenarios.finetune.train.runner.LLMFinetuneRunner"  # TODO
    """Code runner.
    Function: Execute LLM fine-tuning code in a Docker environment.
    """

    summarizer: str = "rdagent.scenarios.finetune.dev.feedback.FTExperiment2Feedback"
    """Result summarizer - To be implemented.
    Function: Analyze fine-tuning results and generate feedback, including performance metrics and error analysis.
    """

    # Timeouts (longer for LLM training, all for Docker container timeout)
    full_timeout: int = 360000
    """Full training timeout in seconds (default 100 hours, env: FT_FULL_TIMEOUT). Used in running stage for complete model training."""
    data_processing_timeout: int = 3600
    """Data processing script timeout in seconds (default 1 hour, env: FT_DATA_PROCESSING_TIMEOUT). Used for full data processing in running stage."""
    debug_data_processing_timeout: int = 1200
    """Debug data processing timeout in seconds (default 20 minutes, env: FT_DEBUG_DATA_PROCESSING_TIMEOUT). Used for --debug mode in coding stage."""
    micro_batch_timeout: int = 1800
    """Micro-batch test timeout in seconds (default 30 minutes, env: FT_MICRO_BATCH_TIMEOUT)."""

    # Pipeline behavior
    coder_on_whole_pipeline: bool = True
    app_tpl: str = "scenarios/finetune"

    # Benchmark evaluation (always enabled as part of evaluation pipeline)

    benchmark_timeout: int = 0
    """Benchmark evaluation timeout in seconds. 0 means no timeout."""

    # Judge API configuration (for llmjudge benchmarks like AIME)
    judge_model: str = "gpt-5.1"
    """LLM judge model name for evaluation"""

    judge_api_key: str | None = None
    """API key for judge model (if None, will try to use from environment)"""

    judge_api_base: str | None = None
    """API base URL for judge model (if None, will use default)"""

    judge_retry: int = 10
    """Number of retries for LLM judge API calls (env: FT_JUDGE_RETRY)"""

    benchmark_limit: int | None = None
    """Limit number of samples for benchmark evaluation (None for full evaluation). Use for quick testing and debugging."""

    benchmark_num_runs: int = 1
    """Number of times to run each sample (for computing average or pass@k). Set >1 for multiple runs."""

    benchmark_pass_k: list[int] | None = None
    """Pass@k parameter list for code generation tasks (e.g., [1, 5, 10]). None to disable."""

    # Data paths and processing
    file_path: Path = Path.cwd() / "git_ignore_folder" / "finetune_files"
    show_nan_columns: bool = False
    sample_data_by_LLM: bool = True

    # LLM-specific fields
    user_target_scenario: str | None = None
    target_benchmark: str | None = None
    """Benchmark dataset to evaluate on. Supported: aime25, aime24, mmlu, gsm8k, math, etc."""
    benchmark_description: str | None = None
    base_model: str | None = None
    dataset: str | None = None
    upper_data_size_limit: int = 2000

    # Data processing LLM models (for API calls in data processing scripts)
    strong_models: list[str] = ["gpt-5", "gpt-5.1"]
    """Strong models for complex tasks (CoT generation, reasoning) - supports list (env: FT_STRONG_MODELS)"""

    weak_models: list[str] = ["gpt-4o-mini", "o4-mini", "gpt-5-mini"]
    """Weak models for simple tasks (filtering, format conversion) - supports list (env: FT_WEAK_MODELS)"""

    embedding_models: list[str] = ["text-embedding-3-small", "text-embedding-3-large"]

    # Docker settings
    docker_enable_cache: bool = False
    """Enable Docker cache for training (set via FT_DOCKER_ENABLE_CACHE)"""

    # data sample count
    data_sample_count: int = 3

    # API concurrency for data processing
    api_max_workers: int = 1000
    """Max concurrent workers for LLM API calls in data processing scripts (env: FT_API_MAX_WORKERS)"""

    # Coder settings
    coder_max_loop: int = 10

    # CoT format settings
    force_think_token: bool = False
    """Force <think> token wrapping for CoT training data (env: FT_FORCE_THINK_TOKEN).
    When True: Data must be wrapped in <think>...</think> format, benchmark uses extract-non-reasoning-content postprocessor.
    When False: CoT reasoning required but format is flexible, no postprocessor needed."""


# Global setting instance for LLM finetuning scenario
FT_RD_SETTING = LLMFinetunePropSetting()
